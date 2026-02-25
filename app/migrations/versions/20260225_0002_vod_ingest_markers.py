"""Add VOD ingest completion markers

Revision ID: 20260225_0002
Revises: 20260222_0001
Create Date: 2026-02-25 00:00:00
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260225_0002"
down_revision: Union[str, None] = "20260222_0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE `vod_ingest_markers` (
          `vod_id` BIGINT UNSIGNED NOT NULL,
          `source_filename` VARCHAR(255) NOT NULL,
          `source_file_sha256` CHAR(64) NOT NULL,
          `source_file_size` BIGINT UNSIGNED NOT NULL,
          `comments_ingested` INT UNSIGNED NOT NULL,
          `completed_at` DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
          `updated_at` DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
          PRIMARY KEY (`vod_id`),
          KEY `idx_vod_ingest_markers_sha` (`source_file_sha256`),
          CONSTRAINT `fk_vod_ingest_markers_vod`
            FOREIGN KEY (`vod_id`) REFERENCES `vods` (`vod_id`)
            ON DELETE CASCADE ON UPDATE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS `vod_ingest_markers`")
