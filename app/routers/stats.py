from collections import defaultdict

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse
from scipy.stats import mannwhitneyu
from sqlalchemy import text

from core.config import DEFAULT_PLATFORM
from core.db import SessionLocal
from core.templates import templates

router = APIRouter()

@router.get("/u/{login}/stats", response_class=HTMLResponse)
def user_stats_page(
    request: Request,
    login: str,
    platform: str = Query(DEFAULT_PLATFORM),
):
    with SessionLocal() as db:
        # user lookup
        user_row = db.execute(
            text("""
                SELECT user_id, login, display_name
                FROM users
                WHERE platform = :platform AND login = :login
                LIMIT 1
            """),
            {"platform": platform, "login": login},
        ).mappings().first()

        if not user_row:
            return templates.TemplateResponse(
                "user_stats.html",
                {
                    "request": request,
                    "error": f"ユーザが見つかりませんでした: {platform}/{login}",
                    "user": None,
                    "stats": [],
                    "owners_stats": [],
                    "owners_total_comments": 0,
                    "impact_stats": [],
                    "impact_total": None,
                    "cn_scores": None,
                    "cn_status_dist": {},
                },
                status_code=404,
            )

        uid = user_row["user_id"]

        total_comments_row = db.execute(
            text("""
                SELECT COUNT(*) AS cnt
                FROM comments
                WHERE commenter_user_id = :uid
            """),
            {"uid": uid},
        ).mappings().first()
        owners_total_comments = int(total_comments_row["cnt"] or 0)

        # Aggregate comment times by hour (JST)
        stats_rows = db.execute(
            text("""
                SELECT HOUR(comment_created_at_utc + INTERVAL 9 HOUR) AS hour, COUNT(*) AS count
                FROM comments
                WHERE commenter_user_id = :uid AND comment_created_at_utc IS NOT NULL
                GROUP BY hour
                ORDER BY hour
            """),
            {"uid": uid},
        ).mappings().all()

        # Prepare stats for 24 hours
        stats = [0] * 24
        for row in stats_rows:
            hour = row["hour"]
            if 0 <= hour < 24:
                stats[hour] = row["count"]

        # Aggregate comment times by weekday (JST)
        weekday_rows = db.execute(
            text("""
                SELECT DAYOFWEEK(comment_created_at_utc + INTERVAL 9 HOUR) AS weekday, COUNT(*) AS count
                FROM comments
                WHERE commenter_user_id = :uid AND comment_created_at_utc IS NOT NULL
                GROUP BY weekday
                ORDER BY weekday
            """),
            {"uid": uid},
        ).mappings().all()

        # Prepare weekday stats (0=Sun, 1=Mon, ..., 6=Sat)
        weekday_stats = [0] * 7
        for row in weekday_rows:
            wd = row["weekday"] - 1  # DAYOFWEEK: 1=Sun, 2=Mon, ..., 7=Sat -> 0=Sun, ..., 6=Sat
            if 0 <= wd < 7:
                weekday_stats[wd] = row["count"]

        # Aggregate comments by owner
        owners_rows = db.execute(
            text("""
                SELECT u.user_id AS owner_user_id, u.login, u.display_name, COUNT(*) AS count
                FROM comments c
                JOIN vods v ON v.vod_id = c.vod_id
                JOIN users u ON u.user_id = v.owner_user_id
                WHERE c.commenter_user_id = :uid
                GROUP BY u.user_id, u.login, u.display_name
                ORDER BY count DESC
                LIMIT 50
            """),
            {"uid": uid},
        ).mappings().all()

        owner_activity_rows = db.execute(
            text("""
                WITH target_vods AS (
                    SELECT DISTINCT c.vod_id
                    FROM comments c
                    WHERE c.commenter_user_id = :uid
                ),
                vod_totals AS (
                    SELECT
                        v.owner_user_id,
                        v.vod_id,
                        GREATEST(
                            COALESCE(CEIL(v.length_seconds / 300.0), 0),
                            COALESCE(CEIL((MAX(ca.offset_seconds) + 1) / 300.0), 0),
                            1
                        ) AS total_buckets
                    FROM target_vods tv
                    JOIN vods v ON v.vod_id = tv.vod_id
                    LEFT JOIN comments ca ON ca.vod_id = tv.vod_id
                    GROUP BY v.owner_user_id, v.vod_id, v.length_seconds
                ),
                owner_totals AS (
                    SELECT owner_user_id, SUM(total_buckets) AS total_buckets
                    FROM vod_totals
                    GROUP BY owner_user_id
                ),
                owner_active AS (
                    SELECT
                        v.owner_user_id,
                        COUNT(DISTINCT CONCAT(c.vod_id, ':', FLOOR(c.offset_seconds / 300))) AS active_buckets
                    FROM comments c
                    JOIN vods v ON v.vod_id = c.vod_id
                    WHERE c.commenter_user_id = :uid
                    GROUP BY v.owner_user_id
                )
                SELECT
                    ot.owner_user_id,
                    ot.total_buckets,
                    COALESCE(oa.active_buckets, 0) AS active_buckets
                FROM owner_totals ot
                LEFT JOIN owner_active oa ON oa.owner_user_id = ot.owner_user_id
            """),
            {"uid": uid},
        ).mappings().all()
        owner_activity_map = {int(row["owner_user_id"]): row for row in owner_activity_rows}

        owners_stats = []
        for idx, row in enumerate(owners_rows, start=1):
            owner_id = int(row["owner_user_id"])
            count = int(row["count"] or 0)
            ratio = round((count / owners_total_comments) * 100, 1) if owners_total_comments > 0 else 0.0
            activity_row = owner_activity_map.get(owner_id)
            active_rate = None
            inactive_rate = None
            if activity_row:
                total_buckets = int(activity_row["total_buckets"] or 0)
                active_buckets = int(activity_row["active_buckets"] or 0)
                if total_buckets > 0:
                    active_rate = round((active_buckets / total_buckets) * 100, 1)
                    inactive_rate = round(100.0 - active_rate, 1)
            owners_stats.append({
                "rank": idx,
                "login": row["login"],
                "display_name": row["display_name"],
                "count": count,
                "ratio": ratio,
                "active_rate": active_rate,
                "inactive_rate": inactive_rate,
            })

        # ---- コミュニティノート平均スコア ----
        cn_avg = db.execute(
            text("""
                SELECT
                    AVG(cn.verifiability) AS avg_verifiability,
                    AVG(cn.harm_risk) AS avg_harm_risk,
                    AVG(cn.exaggeration) AS avg_exaggeration,
                    AVG(cn.evidence_gap) AS avg_evidence_gap,
                    AVG(cn.subjectivity) AS avg_subjectivity,
                    COUNT(*) AS note_count
                FROM community_notes cn
                JOIN comments c ON c.comment_id = cn.comment_id
                WHERE c.commenter_user_id = :uid
            """),
            {"uid": uid},
        ).mappings().first()

        cn_scores = None
        if cn_avg and cn_avg["note_count"] > 0:
            avg_harm = float(cn_avg["avg_harm_risk"] or 0)
            avg_exag = float(cn_avg["avg_exaggeration"] or 0)
            avg_evid = float(cn_avg["avg_evidence_gap"] or 0)
            avg_subj = float(cn_avg["avg_subjectivity"] or 0)
            avg_danger = round((avg_harm + avg_exag + avg_evid + avg_subj) / 4, 1)

            # 危険度の分布（10刻みのヒストグラム）
            danger_dist_rows = db.execute(
                text("""
                    SELECT
                        LEAST(FLOOR((cn.harm_risk + cn.exaggeration + cn.evidence_gap + cn.subjectivity) / 4 / 10) * 10, 90) AS bucket,
                        COUNT(*) AS cnt
                    FROM community_notes cn
                    JOIN comments c ON c.comment_id = cn.comment_id
                    WHERE c.commenter_user_id = :uid
                    GROUP BY bucket
                    ORDER BY bucket
                """),
                {"uid": uid},
            ).mappings().all()
            danger_dist = [0] * 10  # 0-9, 10-19, ..., 90-100
            for row in danger_dist_rows:
                idx = int(row["bucket"]) // 10
                if 0 <= idx < 10:
                    danger_dist[idx] = row["cnt"]

            cn_scores = {
                "avg_verifiability": round(float(cn_avg["avg_verifiability"] or 0), 1),
                "avg_harm_risk": round(avg_harm, 1),
                "avg_exaggeration": round(avg_exag, 1),
                "avg_evidence_gap": round(avg_evid, 1),
                "avg_subjectivity": round(avg_subj, 1),
                "avg_danger": avg_danger,
                "note_count": cn_avg["note_count"],
                "danger_dist": danger_dist,
            }

        # ステータス分布
        cn_status_rows = db.execute(
            text("""
                SELECT cn.status, COUNT(*) AS cnt
                FROM community_notes cn
                JOIN comments c ON c.comment_id = cn.comment_id
                WHERE c.commenter_user_id = :uid
                GROUP BY cn.status
            """),
            {"uid": uid},
        ).mappings().all()
        cn_status_dist = {row["status"]: row["cnt"] for row in cn_status_rows}

        # ---- コメント影響度分析 ----
        vod_count = db.execute(
            text("SELECT COUNT(DISTINCT vod_id) AS cnt FROM comments WHERE commenter_user_id = :uid"),
            {"uid": uid},
        ).mappings().first()["cnt"]

        impact_stats = []
        impact_total = None
        if vod_count <= 500:
            # 生バケットデータを取得（統計検定のため個別行が必要）
            bucket_rows = db.execute(
                text("""
                    WITH target_vods AS (
                        SELECT DISTINCT vod_id FROM comments WHERE commenter_user_id = :uid
                    )
                    SELECT
                        v.owner_user_id,
                        u.login AS owner_login,
                        u.display_name AS owner_display_name,
                        c.vod_id,
                        FLOOR(c.offset_seconds / 300) AS bucket,
                        SUM(CASE WHEN c.commenter_user_id != :uid THEN 1 ELSE 0 END) AS other_comments,
                        COUNT(DISTINCT CASE WHEN c.commenter_user_id != :uid THEN c.commenter_user_id END) AS other_unique,
                        MAX(CASE WHEN c.commenter_user_id = :uid THEN 1 ELSE 0 END) AS target_active
                    FROM comments c
                    INNER JOIN target_vods tv ON tv.vod_id = c.vod_id
                    JOIN vods v ON v.vod_id = c.vod_id
                    JOIN users u ON u.user_id = v.owner_user_id
                    GROUP BY v.owner_user_id, u.login, u.display_name, c.vod_id, bucket
                    HAVING other_comments > 0
                """),
                {"uid": uid},
            ).mappings().all()

            # オーナー別にバケットデータを振り分け
            owner_buckets = defaultdict(lambda: {
                "login": "", "display_name": "",
                "active_comments": [], "inactive_comments": [],
                "active_unique": [], "inactive_unique": [],
            })
            all_active_comments = []
            all_inactive_comments = []
            all_active_unique = []
            all_inactive_unique = []

            for row in bucket_rows:
                oid = row["owner_user_id"]
                d = owner_buckets[oid]
                d["login"] = row["owner_login"]
                d["display_name"] = row["owner_display_name"] or row["owner_login"]
                oc = float(row["other_comments"])
                ou = float(row["other_unique"])
                if row["target_active"]:
                    d["active_comments"].append(oc)
                    d["active_unique"].append(ou)
                    all_active_comments.append(oc)
                    all_active_unique.append(ou)
                else:
                    d["inactive_comments"].append(oc)
                    d["inactive_unique"].append(ou)
                    all_inactive_comments.append(oc)
                    all_inactive_unique.append(ou)

            def calc_impact(active, inactive):
                """平均、変化率、Mann-Whitney U検定のp値を計算"""
                avg_a = round(sum(active) / len(active), 2)
                avg_i = round(sum(inactive) / len(inactive), 2)
                change = round((avg_a - avg_i) / avg_i * 100, 1) if avg_i > 0 else 0.0
                _, p = mannwhitneyu(active, inactive, alternative='two-sided')
                return avg_a, avg_i, change, round(p, 4)

            # オーナー別に統計を計算
            for oid, d in sorted(owner_buckets.items(), key=lambda x: len(x[1]["active_comments"]), reverse=True):
                if len(d["active_comments"]) < 3 or len(d["inactive_comments"]) < 3:
                    continue
                avg_a, avg_i, c_change, c_p = calc_impact(d["active_comments"], d["inactive_comments"])
                avg_ua, avg_ui, u_change, u_p = calc_impact(d["active_unique"], d["inactive_unique"])
                impact_stats.append({
                    "owner_login": d["login"],
                    "owner_display_name": d["display_name"],
                    "active_buckets": len(d["active_comments"]),
                    "inactive_buckets": len(d["inactive_comments"]),
                    "avg_others_active": avg_a,
                    "avg_others_inactive": avg_i,
                    "comment_change": c_change,
                    "p_value": c_p,
                    "avg_unique_active": avg_ua,
                    "avg_unique_inactive": avg_ui,
                    "unique_change": u_change,
                    "p_value_unique": u_p,
                })

            # 合計行
            if len(all_active_comments) >= 3 and len(all_inactive_comments) >= 3:
                avg_a, avg_i, c_change, c_p = calc_impact(all_active_comments, all_inactive_comments)
                avg_ua, avg_ui, u_change, u_p = calc_impact(all_active_unique, all_inactive_unique)
                impact_total = {
                    "active_buckets": len(all_active_comments),
                    "inactive_buckets": len(all_inactive_comments),
                    "avg_others_active": avg_a,
                    "avg_others_inactive": avg_i,
                    "comment_change": c_change,
                    "p_value": c_p,
                    "avg_unique_active": avg_ua,
                    "avg_unique_inactive": avg_ui,
                    "unique_change": u_change,
                    "p_value_unique": u_p,
                }

        return templates.TemplateResponse(
            "user_stats.html",
            {
                "request": request,
                "error": None,
                "user": dict(user_row),
                "stats": stats,
                "weekday_stats": weekday_stats,
                "owners_stats": owners_stats,
                "owners_total_comments": owners_total_comments,
                "impact_stats": impact_stats,
                "impact_total": impact_total,
                "cn_scores": cn_scores,
                "cn_status_dist": cn_status_dist,
                "platform": platform,
            },
        )


