-- Metadata table to store database version and creation timestamp
CREATE TABLE IF NOT EXISTS [DatabaseMetadata] (
	[ID] INTEGER PRIMARY KEY CHECK ([ID] = 1),
	[CreatedTimestamp] TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	[Version] TEXT DEFAULT (strftime('%Y%m%d', 'now')),
	[TotalInformants] INTEGER NULL,
	[Notes] TEXT NULL
);

-- Insert initial metadata record
INSERT OR REPLACE INTO [DatabaseMetadata] ([ID], [CreatedTimestamp], [Version]) 
VALUES (1, CURRENT_TIMESTAMP, strftime('%Y%m%d', 'now'));
