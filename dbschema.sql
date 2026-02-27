-- Adminer 5.4.1 MySQL 8.0.45 dump

SET NAMES utf8;
SET time_zone = '+00:00';
SET foreign_key_checks = 0;
SET sql_mode = 'NO_AUTO_VALUE_ON_ZERO';

SET NAMES utf8mb4;

CREATE TABLE `comments` (
  `comment_id` varchar(128) NOT NULL,
  `vod_id` bigint unsigned NOT NULL,
  `offset_seconds` int unsigned NOT NULL,
  `comment_created_at_utc` datetime(6) DEFAULT NULL,
  `commenter_user_id` bigint unsigned DEFAULT NULL,
  `commenter_login_snapshot` varchar(64) DEFAULT NULL,
  `commenter_display_name_snapshot` varchar(128) DEFAULT NULL,
  `body` text NOT NULL,
  `community_note_body` text,
  `community_note_created_at_utc` datetime(6) DEFAULT NULL,
  `community_note_updated_at_utc` datetime(6) DEFAULT NULL,
  `user_color` varchar(16) DEFAULT NULL,
  `bits_spent` int unsigned DEFAULT NULL,
  `raw_json` json DEFAULT NULL,
  `ingested_at` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  `twicome_likes_count` int unsigned NOT NULL DEFAULT '0',
  `twicome_dislikes_count` int unsigned NOT NULL DEFAULT '0',
  PRIMARY KEY (`comment_id`),
  KEY `idx_comments_vod_time` (`vod_id`,`offset_seconds`),
  KEY `idx_comments_user_vod_time` (`commenter_user_id`,`vod_id`,`offset_seconds`),
  KEY `idx_comments_created_at` (`comment_created_at_utc`),
  KEY `idx_comments_note_created_at` (`community_note_created_at_utc`),
  KEY `idx_comments_user_created` (`commenter_user_id`,`comment_created_at_utc`),
  KEY `idx_comments_vod_offset_user` (`vod_id`,`offset_seconds`,`commenter_user_id`),
  KEY `idx_comments_score` (((`twicome_likes_count` + `twicome_dislikes_count`))),
  KEY `idx_comments_commenter_login_at` (`commenter_login_snapshot`,`comment_created_at_utc`),
  KEY `idx_comments_commenter_login_vod` (`commenter_login_snapshot`,`vod_id`),
  CONSTRAINT `fk_comments_commenter` FOREIGN KEY (`commenter_user_id`) REFERENCES `users` (`user_id`) ON DELETE SET NULL ON UPDATE CASCADE,
  CONSTRAINT `fk_comments_vod` FOREIGN KEY (`vod_id`) REFERENCES `vods` (`vod_id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;


CREATE TABLE `community_notes` (
  `note_id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `comment_id` varchar(128) NOT NULL,
  `eligible` tinyint(1) NOT NULL,
  `status` enum('supported','insufficient','inconsistent','not_applicable') NOT NULL,
  `note` text NOT NULL,
  `verifiability` tinyint unsigned NOT NULL,
  `harm_risk` tinyint unsigned NOT NULL,
  `exaggeration` tinyint unsigned NOT NULL,
  `evidence_gap` tinyint unsigned NOT NULL,
  `subjectivity` tinyint unsigned NOT NULL DEFAULT '0',
  `issues` json DEFAULT NULL,
  `ask` varchar(255) NOT NULL DEFAULT '',
  `note_json` json NOT NULL,
  `model` varchar(64) DEFAULT NULL,
  `prompt_version` varchar(32) DEFAULT NULL,
  `created_at_utc` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  `updated_at_utc` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (`note_id`),
  UNIQUE KEY `uq_community_notes_comment` (`comment_id`),
  KEY `idx_community_notes_status` (`status`),
  KEY `idx_community_notes_harm` (`harm_risk`),
  CONSTRAINT `fk_community_notes_comment` FOREIGN KEY (`comment_id`) REFERENCES `comments` (`comment_id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;


CREATE TABLE `users` (
  `user_id` bigint unsigned NOT NULL,
  `login` varchar(64) NOT NULL,
  `display_name` varchar(128) DEFAULT NULL,
  `profile_image_url` varchar(512) DEFAULT NULL,
  `platform` varchar(32) NOT NULL DEFAULT 'twitch',
  `created_at` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  `updated_at` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (`user_id`),
  UNIQUE KEY `uq_users_platform_login` (`platform`,`login`),
  KEY `idx_users_vod_fetch` (`platform`,`user_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;


CREATE TABLE `vods` (
  `vod_id` bigint unsigned NOT NULL,
  `owner_user_id` bigint unsigned NOT NULL,
  `title` varchar(512) NOT NULL,
  `description` text,
  `created_at_utc` datetime(6) DEFAULT NULL,
  `length_seconds` int unsigned DEFAULT NULL,
  `start_seconds` int unsigned DEFAULT NULL,
  `end_seconds` int unsigned DEFAULT NULL,
  `view_count` int unsigned DEFAULT NULL,
  `game_name` varchar(255) DEFAULT NULL,
  `platform` varchar(32) NOT NULL DEFAULT 'twitch',
  `url` varchar(512) DEFAULT NULL,
  `youtube_url` varchar(512) DEFAULT NULL,
  `ingested_at` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  PRIMARY KEY (`vod_id`),
  KEY `idx_vods_owner` (`owner_user_id`),
  CONSTRAINT `fk_vods_owner` FOREIGN KEY (`owner_user_id`) REFERENCES `users` (`user_id`) ON DELETE RESTRICT ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;


CREATE TABLE `vod_ingest_markers` (
  `vod_id` bigint unsigned NOT NULL,
  `source_filename` varchar(255) NOT NULL,
  `source_file_sha256` char(64) NOT NULL,
  `source_file_size` bigint unsigned NOT NULL,
  `comments_ingested` int unsigned NOT NULL,
  `completed_at` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  `updated_at` datetime(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (`vod_id`),
  KEY `idx_vod_ingest_markers_sha` (`source_file_sha256`),
  CONSTRAINT `fk_vod_ingest_markers_vod` FOREIGN KEY (`vod_id`) REFERENCES `vods` (`vod_id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;


-- 2026-02-18 10:14:43 UTC
