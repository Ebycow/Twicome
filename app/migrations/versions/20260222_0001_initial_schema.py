"""Initial MySQL schema

Revision ID: 20260222_0001
Revises:
Create Date: 2026-02-22 00:00:00
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260222_0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE `users` (
          `user_id` BIGINT UNSIGNED NOT NULL,
          `login` VARCHAR(64) NOT NULL,
          `display_name` VARCHAR(128) DEFAULT NULL,
          `profile_image_url` VARCHAR(512) DEFAULT NULL,
          `platform` VARCHAR(32) NOT NULL DEFAULT 'twitch',
          `created_at` DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
          `updated_at` DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
          PRIMARY KEY (`user_id`),
          UNIQUE KEY `uq_users_platform_login` (`platform`, `login`),
          KEY `idx_users_vod_fetch` (`platform`, `user_id`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
        """
    )

    op.execute(
        """
        CREATE TABLE `vods` (
          `vod_id` BIGINT UNSIGNED NOT NULL,
          `owner_user_id` BIGINT UNSIGNED NOT NULL,
          `title` VARCHAR(512) NOT NULL,
          `description` TEXT,
          `created_at_utc` DATETIME(6) DEFAULT NULL,
          `length_seconds` INT UNSIGNED DEFAULT NULL,
          `start_seconds` INT UNSIGNED DEFAULT NULL,
          `end_seconds` INT UNSIGNED DEFAULT NULL,
          `view_count` INT UNSIGNED DEFAULT NULL,
          `game_name` VARCHAR(255) DEFAULT NULL,
          `platform` VARCHAR(32) NOT NULL DEFAULT 'twitch',
          `url` VARCHAR(512) DEFAULT NULL,
          `youtube_url` VARCHAR(512) DEFAULT NULL,
          `ingested_at` DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
          PRIMARY KEY (`vod_id`),
          KEY `idx_vods_owner` (`owner_user_id`),
          CONSTRAINT `fk_vods_owner` FOREIGN KEY (`owner_user_id`) REFERENCES `users` (`user_id`) ON DELETE RESTRICT ON UPDATE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
        """
    )

    op.execute(
        """
        CREATE TABLE `comments` (
          `comment_id` VARCHAR(128) NOT NULL,
          `vod_id` BIGINT UNSIGNED NOT NULL,
          `offset_seconds` INT UNSIGNED NOT NULL,
          `comment_created_at_utc` DATETIME(6) DEFAULT NULL,
          `commenter_user_id` BIGINT UNSIGNED DEFAULT NULL,
          `commenter_login_snapshot` VARCHAR(64) DEFAULT NULL,
          `commenter_display_name_snapshot` VARCHAR(128) DEFAULT NULL,
          `body` TEXT NOT NULL,
          `community_note_body` TEXT,
          `community_note_created_at_utc` DATETIME(6) DEFAULT NULL,
          `community_note_updated_at_utc` DATETIME(6) DEFAULT NULL,
          `user_color` VARCHAR(16) DEFAULT NULL,
          `bits_spent` INT UNSIGNED DEFAULT NULL,
          `raw_json` JSON DEFAULT NULL,
          `ingested_at` DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
          `twicome_likes_count` INT UNSIGNED NOT NULL DEFAULT '0',
          `twicome_dislikes_count` INT UNSIGNED NOT NULL DEFAULT '0',
          PRIMARY KEY (`comment_id`),
          KEY `idx_comments_vod_time` (`vod_id`, `offset_seconds`),
          KEY `idx_comments_user_vod_time` (`commenter_user_id`, `vod_id`, `offset_seconds`),
          KEY `idx_comments_created_at` (`comment_created_at_utc`),
          KEY `idx_comments_note_created_at` (`community_note_created_at_utc`),
          KEY `idx_comments_user_created` (`commenter_user_id`, `comment_created_at_utc`),
          KEY `idx_comments_vod_offset_user` (`vod_id`, `offset_seconds`, `commenter_user_id`),
          KEY `idx_comments_score` (((`twicome_likes_count` + `twicome_dislikes_count`))),
          CONSTRAINT `fk_comments_commenter` FOREIGN KEY (`commenter_user_id`) REFERENCES `users` (`user_id`) ON DELETE SET NULL ON UPDATE CASCADE,
          CONSTRAINT `fk_comments_vod` FOREIGN KEY (`vod_id`) REFERENCES `vods` (`vod_id`) ON DELETE CASCADE ON UPDATE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
        """
    )

    op.execute(
        """
        CREATE TABLE `community_notes` (
          `note_id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
          `comment_id` VARCHAR(128) NOT NULL,
          `eligible` TINYINT(1) NOT NULL,
          `status` ENUM('supported', 'insufficient', 'inconsistent', 'not_applicable') NOT NULL,
          `note` TEXT NOT NULL,
          `verifiability` TINYINT UNSIGNED NOT NULL,
          `harm_risk` TINYINT UNSIGNED NOT NULL,
          `exaggeration` TINYINT UNSIGNED NOT NULL,
          `evidence_gap` TINYINT UNSIGNED NOT NULL,
          `subjectivity` TINYINT UNSIGNED NOT NULL DEFAULT '0',
          `issues` JSON DEFAULT NULL,
          `ask` VARCHAR(255) NOT NULL DEFAULT '',
          `note_json` JSON NOT NULL,
          `model` VARCHAR(64) DEFAULT NULL,
          `prompt_version` VARCHAR(32) DEFAULT NULL,
          `created_at_utc` DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
          `updated_at_utc` DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
          PRIMARY KEY (`note_id`),
          UNIQUE KEY `uq_community_notes_comment` (`comment_id`),
          KEY `idx_community_notes_status` (`status`),
          KEY `idx_community_notes_harm` (`harm_risk`),
          CONSTRAINT `fk_community_notes_comment` FOREIGN KEY (`comment_id`) REFERENCES `comments` (`comment_id`) ON DELETE CASCADE ON UPDATE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS `community_notes`")
    op.execute("DROP TABLE IF EXISTS `comments`")
    op.execute("DROP TABLE IF EXISTS `vods`")
    op.execute("DROP TABLE IF EXISTS `users`")
