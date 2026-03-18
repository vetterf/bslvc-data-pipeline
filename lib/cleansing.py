#!/usr/bin/env python3
"""
Unified Data Normalization System
=================================

This script combines all data normalization tasks into a single unified system:
1. Gender normalization
2. School type normalization (Primary & Secondary)
3. Qualification mapping
4. Language normalization

All mapping files are stored in the 'mappings/' subfolder for better organization.

The script provides two main operations:
1. UPDATE MODE: Generate/update CSV mapping files for manual review and editing
2. APPLY MODE: Apply the CSV mappings to the database

Usage:
    python unified_data_normalization.py --mode update    # Generate/update CSV mapping files
    python unified_data_normalization.py --mode apply     # Apply mappings to database
    python unified_data_normalization.py --help           # Show help

Features:
- Unified configuration management
- Manual override capability via CSV files
- Comprehensive statistics and verification
- Error handling and progress tracking
- Backup functionality for database safety
"""

import argparse
import sqlite3
import pandas as pd
import os
import re
import sys
from datetime import datetime
from collections import Counter, defaultdict

from lib import DB_PATH, MAPPINGS_DIR

# Flair imports are now handled within the language processing function
# to make them optional dependencies

# Configuration – all paths derived from the shared constants in lib/__init__.py
_M = str(MAPPINGS_DIR)
CONFIG = {
    'database': {
        'path': str(DB_PATH),
        'table': 'Informants',
        'backup': True
    },
    'columns': {
        'gender': 'Gender',
        'primary_school': 'PrimarySchool',
        'secondary_school': 'SecondarySchool',
        'qualifications': ['Qualifications', 'QualiMother', 'QualiFather', 'QualiPartner'],
        'languages': ['LanguageHome', 'LanguageFather', 'LanguageMother']
    },
    'output_files': {
        'gender': os.path.join(_M, 'gender_mapping.csv'),
        'primary_school': os.path.join(_M, 'primary_school_mapping.csv'),
        'secondary_school': os.path.join(_M, 'secondary_school_mapping.csv'),
        'qualifications': os.path.join(_M, 'qualification_mapping.csv'),
        'languages': os.path.join(_M, 'language_mapping.csv')
    },
    'manual_override_files': {
        'gender': os.path.join(_M, 'gender_mapping_manual.csv'),
        'primary_school': os.path.join(_M, 'primary_school_mapping_manual.csv'),
        'secondary_school': os.path.join(_M, 'secondary_school_mapping_manual.csv'),
        'qualifications': os.path.join(_M, 'qualification_mapping_manual.csv'),
        'qualifications_by_id': os.path.join(_M, 'qualification_mapping_manual_InformantID_Occupation.csv'),
        'languages': os.path.join(_M, 'language_mapping_manual.csv')
    }
}

class UnifiedDataNormalizer:
    """
    Unified Data Normalization System for standardizing survey data.
    """
    
    def __init__(self, config=CONFIG):
        self.config = config
        self.db_path = config['database']['path']
        self.table_name = config['database']['table']
        
        # Ensure mappings directory exists
        mappings_dir = str(MAPPINGS_DIR)
        if not os.path.exists(mappings_dir):
            os.makedirs(mappings_dir)
            print(f"Created mappings directory: {mappings_dir}")
        
        # Statistics tracking
        self.stats = {
            'gender': {},
            'schools': {},
            'qualifications': {},
            'languages': {}
        }
    
    def backup_database(self):
        """Create a backup of the database before making changes."""
        if not self.config['database']['backup']:
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{self.db_path}.backup_{timestamp}"
            
            print(f"Creating database backup: {backup_path}")
            
            # Simple file copy for SQLite
            import shutil
            shutil.copy2(self.db_path, backup_path)
            
            print(f"✅ Database backed up successfully")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not create database backup: {e}")
            response = input("Continue without backup? (y/N): ")
            if response.lower() != 'y':
                sys.exit(1)
    
    def sync_manual_mapping_files(self):
        """
        Compare auto-generated mapping files with manual mapping files.
        Add new values from auto-generated files to manual files, preserving existing manual mappings.
        """
        print("\n🔄 SYNCING MANUAL MAPPING FILES")
        print("=" * 80)
        print("Checking for new values to add to manual mapping files...")
        print()
        
        sync_configs = [
            {
                'name': 'Gender',
                'auto_file': self.config['output_files']['gender'],
                'manual_file': self.config['manual_override_files']['gender'],
                'original_col': 'Original_Gender',
                'mapped_col': 'Normalized_Gender'
            },
            {
                'name': 'Primary School',
                'auto_file': self.config['output_files']['primary_school'],
                'manual_file': self.config['manual_override_files']['primary_school'],
                'original_col': 'Original_School',
                'mapped_col': 'Normalized_School'
            },
            {
                'name': 'Secondary School',
                'auto_file': self.config['output_files']['secondary_school'],
                'manual_file': self.config['manual_override_files']['secondary_school'],
                'original_col': 'Original_School',
                'mapped_col': 'Normalized_School'
            },
            {
                'name': 'Qualifications',
                'auto_file': self.config['output_files']['qualifications'],
                'manual_file': self.config['manual_override_files']['qualifications'],
                'original_col': 'Original_Qualification',
                'mapped_col': 'Mapped_Category'
            },
            {
                'name': 'Languages',
                'auto_file': self.config['output_files']['languages'],
                'manual_file': self.config['manual_override_files']['languages'],
                'original_col': 'Original_Language',
                'mapped_col': 'Normalized_Language'
            }
        ]
        
        total_new_values = 0
        
        for config in sync_configs:
            name = config['name']
            auto_file = config['auto_file']
            manual_file = config['manual_file']
            original_col = config['original_col']
            mapped_col = config['mapped_col']
            
            print(f"📋 {name}")
            print(f"   Auto file: {auto_file}")
            print(f"   Manual file: {manual_file}")
            
            # Check if auto file exists
            if not os.path.exists(auto_file):
                print(f"   ⚠️  Auto-generated file not found. Skipping...")
                print()
                continue
            
            # Load auto-generated mappings
            try:
                auto_df = pd.read_csv(auto_file, keep_default_na=False)
                
                # Check if required columns exist
                if original_col not in auto_df.columns or mapped_col not in auto_df.columns:
                    print(f"   ⚠️  Required columns not found in auto file. Skipping...")
                    print()
                    continue
                
                auto_values = set(auto_df[original_col].astype(str))
                print(f"   Auto-generated values: {len(auto_values)}")
                
            except Exception as e:
                print(f"   ❌ Error reading auto file: {e}")
                print()
                continue
            
            # Check if manual file exists
            if not os.path.exists(manual_file):
                print(f"   ℹ️  Manual file doesn't exist. Creating with empty mappings...")
                
                # Create a copy with empty mappings
                try:
                    manual_df_new = auto_df.copy()
                    manual_df_new[mapped_col] = ''  # Set all mappings to empty
                    manual_df_new.to_csv(manual_file, index=False)
                    print(f"   ✅ Created manual file with {len(manual_df_new)} entries (all mappings empty)")
                    total_new_values += len(manual_df_new)
                except Exception as e:
                    print(f"   ❌ Error creating manual file: {e}")
                
                print()
                continue
            
            # Load manual mappings
            try:
                manual_df = pd.read_csv(manual_file, keep_default_na=False)
                
                # Check if required columns exist
                if original_col not in manual_df.columns or mapped_col not in manual_df.columns:
                    print(f"   ⚠️  Required columns not found in manual file. Skipping...")
                    print()
                    continue
                
                manual_values = set(manual_df[original_col].astype(str))
                print(f"   Manual file values: {len(manual_values)}")
                
            except Exception as e:
                print(f"   ❌ Error reading manual file: {e}")
                print()
                continue
            
            # Find new values (in auto but not in manual)
            new_values = auto_values - manual_values
            
            if len(new_values) == 0:
                print(f"   ✅ No new values to add")
                print()
                continue
            
            print(f"   🆕 Found {len(new_values)} new values to add")
            
            # Show some examples of new values
            if len(new_values) <= 5:
                for val in sorted(new_values):
                    print(f"      • {val}")
            else:
                for val in sorted(list(new_values)[:3]):
                    print(f"      • {val}")
                print(f"      ... and {len(new_values) - 3} more")
            
            # Get the new rows from auto_df
            new_rows = auto_df[auto_df[original_col].astype(str).isin(new_values)].copy()
            
            # Set mapped column to empty for new values (makes it clear they need manual review)
            new_rows[mapped_col] = ''
            
            # Append to manual file
            try:
                # Combine and sort
                combined_df = pd.concat([manual_df, new_rows], ignore_index=True)
                
                # Sort by original value for easier manual review
                combined_df = combined_df.sort_values(by=original_col)
                
                # Save back to manual file
                combined_df.to_csv(manual_file, index=False)
                
                print(f"   ✅ Added {len(new_rows)} new entries with empty mappings (requires manual review)")
                print(f"   📊 Manual file now has {len(combined_df)} total entries")
                total_new_values += len(new_rows)
                
            except Exception as e:
                print(f"   ❌ Error updating manual file: {e}")
            
            print()
        
        # Summary
        print("=" * 80)
        if total_new_values > 0:
            print(f"✅ Sync complete! Added {total_new_values} new values with EMPTY mappings")
            print()
            print("⚠️  IMPORTANT:")
            print("   New values have been added with empty mappings to make them easy to identify.")
            print("   You MUST review and fill in the mappings before running apply mode.")
            print()
            print("📝 NEXT STEPS:")
            print("1. Open the manual mapping files in mappings/ folder")
            print("2. Find rows with empty mappings (new values)")
            print("3. Fill in the correct mapping for each new value")
            print("4. Run in apply mode to use the updated manual mappings")
        else:
            print("✅ All manual files are up to date - no new values found")
        
        print()
    
    def check_empty_mappings(self):
        """
        Check all manual mapping files for entries with empty mappings.
        Returns a report of unmapped values that need manual review.
        """
        print("\n🔍 CHECKING FOR UNMAPPED VALUES")
        print("=" * 80)
        
        check_configs = [
            {
                'name': 'Gender',
                'file': self.config['manual_override_files']['gender'],
                'original_col': 'Original_Gender',
                'mapped_col': 'Normalized_Gender'
            },
            {
                'name': 'Primary School',
                'file': self.config['manual_override_files']['primary_school'],
                'original_col': 'Original_School',
                'mapped_col': 'Normalized_School'
            },
            {
                'name': 'Secondary School',
                'file': self.config['manual_override_files']['secondary_school'],
                'original_col': 'Original_School',
                'mapped_col': 'Normalized_School'
            },
            {
                'name': 'Qualifications',
                'file': self.config['manual_override_files']['qualifications'],
                'original_col': 'Original_Qualification',
                'mapped_col': 'Mapped_Category'
            },
            {
                'name': 'Languages',
                'file': self.config['manual_override_files']['languages'],
                'original_col': 'Original_Language',
                'mapped_col': 'Normalized_Language'
            }
        ]
        
        total_empty = 0
        files_with_empty = []
        
        for config in check_configs:
            name = config['name']
            file_path = config['file']
            original_col = config['original_col']
            mapped_col = config['mapped_col']
            
            if not os.path.exists(file_path):
                continue
            
            try:
                df = pd.read_csv(file_path, keep_default_na=False)
                
                if original_col not in df.columns or mapped_col not in df.columns:
                    continue
                
                # Find rows with empty mappings (empty string, NaN, or whitespace only)
                empty_mask = (df[mapped_col].isna() | 
                             (df[mapped_col].astype(str).str.strip() == ''))
                empty_df = df[empty_mask]
                
                if len(empty_df) > 0:
                    print(f"\n📋 {name}")
                    print(f"   File: {file_path}")
                    print(f"   ⚠️  {len(empty_df)} values need mapping:")
                    
                    # Show all empty values (or first 10 if too many)
                    empty_values = empty_df[original_col].tolist()
                    if len(empty_values) <= 10:
                        for val in empty_values:
                            print(f"      • {val}")
                    else:
                        for val in empty_values[:10]:
                            print(f"      • {val}")
                        print(f"      ... and {len(empty_values) - 10} more")
                    
                    total_empty += len(empty_df)
                    files_with_empty.append((name, file_path, len(empty_df)))
                
            except Exception as e:
                print(f"\n❌ Error checking {name}: {e}")
        
        print("\n" + "=" * 80)
        if total_empty > 0:
            print(f"⚠️  WARNING: Found {total_empty} unmapped values across {len(files_with_empty)} file(s)")
            print()
            print("📝 ACTION REQUIRED:")
            print("   Please edit the following files to fill in the empty mappings:")
            for name, file_path, count in files_with_empty:
                print(f"   • {file_path} ({count} values)")
            print()
            print("💡 TIP: Look for rows with empty values in the mapping column.")
            print("   You can use Excel, LibreOffice, or any CSV editor.")
        else:
            print("✅ All manual mapping files are complete - no empty mappings found!")
        
        print()
        return total_empty
    
    def update_all_mappings(self):
        """
        UPDATE MODE: Generate or update all CSV mapping files.
        """
        print("🔄 UNIFIED DATA NORMALIZATION - UPDATE MODE 🔄")
        print("=" * 80)
        print("Generating/updating CSV mapping files for manual review...")
        print()
        
        # 1. Gender normalization
        print("1️⃣  GENDER NORMALIZATION")
        print("-" * 40)
        self.update_gender_mapping()
        print()
        
        # 2. School normalization
        print("2️⃣  SCHOOL NORMALIZATION")
        print("-" * 40)
        self.update_school_mappings()
        print()
        
        # 3. Qualification mapping
        print("3️⃣  QUALIFICATION MAPPING")
        print("-" * 40)
        self.update_qualification_mapping()
        print()
        
        # 4. Language normalization
        print("4️⃣  LANGUAGE NORMALIZATION")
        print("-" * 40)
        self.update_language_mapping()
        print()
        
        # 5. Sync manual mapping files
        print("5️⃣  SYNCING MANUAL MAPPING FILES")
        print("-" * 40)
        self.sync_manual_mapping_files()
        
        # 6. Check for unmapped values
        print("6️⃣  CHECKING FOR UNMAPPED VALUES")
        print("-" * 40)
        self.check_empty_mappings()
        
        # Summary
        self.print_update_summary()
    
    def _setup_id_filter(self, conn, informant_ids):
        """Create a temp table for filtering by InformantID and return a WHERE clause fragment."""
        if informant_ids is None:
            return ""
        cursor = conn.cursor()
        cursor.execute("CREATE TEMP TABLE IF NOT EXISTS _update_ids (InformantID TEXT PRIMARY KEY)")
        cursor.execute("DELETE FROM _update_ids")
        cursor.executemany(
            "INSERT OR IGNORE INTO _update_ids (InformantID) VALUES (?)",
            [(iid,) for iid in informant_ids],
        )
        conn.commit()
        return f" AND {self.table_name}.InformantID IN (SELECT InformantID FROM _update_ids)"

    def _cleanup_id_filter(self, conn):
        """Drop the temp filter table."""
        try:
            conn.execute("DROP TABLE IF EXISTS _update_ids")
        except sqlite3.Error:
            pass

    def apply_all_mappings(self, fill_empty_with_na=False, informant_ids=None):
        """
        APPLY MODE: Apply all CSV mappings to the database.
        
        Args:
            fill_empty_with_na: If True, fill empty mappings with "NA" instead of blocking
            informant_ids: If provided, only normalize these InformantIDs (for incremental updates)
        """
        if informant_ids is not None:
            print("📊 UNIFIED DATA NORMALIZATION - INCREMENTAL APPLY MODE 📊")
            print("=" * 80)
            print(f"Applying CSV mapping files to {len(informant_ids)} new participants...")
        else:
            print("📊 UNIFIED DATA NORMALIZATION - APPLY MODE 📊")
            print("=" * 80)
            print("Applying CSV mapping files to database...")
        print()
        
        if fill_empty_with_na:
            print("⚠️  FILL EMPTY WITH NA MODE ENABLED")
            print("   Empty mappings will be filled with 'NA'")
            print()
        
        # First, check for empty mappings
        print("🔍 PRE-FLIGHT CHECK")
        print("-" * 40)
        empty_count = self.check_empty_mappings()
        
        if empty_count > 0 and not fill_empty_with_na:
            print("\n" + "=" * 80)
            print("❌ CANNOT APPLY MAPPINGS")
            print()
            print("⚠️  You have unmapped values in your manual mapping files.")
            print("   Applying these mappings would result in empty/null values in the database.")
            print()
            print("📝 Please complete the following steps:")
            print("   1. Edit the manual mapping files listed above")
            print("   2. Fill in all empty mappings")
            print("   3. Run apply mode again")
            print()
            print("💡 TIP: You can use the --mode update command to see which values need mapping.")
            print("💡 ALTERNATIVE: Use --fill-empty-with-na flag to fill empty values with 'NA'")
            print()
            return
        
        if empty_count > 0 and fill_empty_with_na:
            print(f"\n⚠️  Found {empty_count} empty mappings - will fill with 'NA'")
            print()
        else:
            print("✅ All manual mapping files are complete")
            print()
        
        # Create backup before making changes
        self.backup_database()
        
        # 1. Apply gender mapping
        print("1️⃣  APPLYING GENDER MAPPING")
        print("-" * 40)
        self.apply_gender_mapping(fill_empty_with_na=fill_empty_with_na, informant_ids=informant_ids)
        print()
        
        # 2. Apply school mappings
        print("2️⃣  APPLYING SCHOOL MAPPINGS")
        print("-" * 40)
        self.apply_school_mappings(fill_empty_with_na=fill_empty_with_na, informant_ids=informant_ids)
        print()
        
        # 3. Apply qualification mapping
        print("3️⃣  APPLYING QUALIFICATION MAPPING")
        print("-" * 40)
        self.apply_qualification_mapping(fill_empty_with_na=fill_empty_with_na, informant_ids=informant_ids)
        print()
        
        # 4. Apply language mapping
        print("4️⃣  APPLYING LANGUAGE MAPPING")
        print("-" * 40)
        self.apply_language_mapping(fill_empty_with_na=fill_empty_with_na, informant_ids=informant_ids)
        print()
        
        # Summary
        self.print_apply_summary()
    
    # ===== GENDER NORMALIZATION =====
    
    def update_gender_mapping(self):
        """Generate gender mapping CSV file."""
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Get unique gender values
            df_gender = pd.read_sql_query(f'''
                SELECT DISTINCT {self.config['columns']['gender']} 
                FROM {self.table_name} 
                ORDER BY {self.config['columns']['gender']}
            ''', conn)
            
            gender_values = df_gender[self.config['columns']['gender']].tolist()
            
            print(f"Found {len(gender_values)} unique gender values in database")
            if len(gender_values) == 0:
                print("❌ No gender values found in database. Check database path and column name.")
                return
            
            # Define normalization patterns
            normalization_patterns = {
                'male_patterns': ['male', 'm', 'man', 'masculine', 'boy', 'gentleman', 'mr', 'sir', 'malte'],
                'female_patterns': ['female', 'f', 'woman', 'feminine', 'girl', 'lady', 'mrs', 'ms', 'miss', 'madam'],
                'nonbinary_patterns': ['non-binary', 'nonbinary', 'non binary', 'nb', 'genderqueer', 'queer', 
                                     'gender fluid', 'genderfluid', 'agender', 'bigender', 'pangender',
                                     'demigender', 'genderless', 'other', 'third gender', 'x'],
                'na_patterns': ['unknown', 'not specified', 'not applicable', 'n/a', 'na', 'null', 
                              'missing', 'unspecified', '?', '-', 'prefer not to say', 'decline to state']
            }
            
            def normalize_gender_value(gender_val):
                if pd.isna(gender_val) or gender_val == '' or gender_val is None:
                    return 'NA'
                
                gender_lower = str(gender_val).lower().strip()
                
                # Check female patterns first to avoid "female" matching "male"
                if any(pattern == gender_lower or (len(pattern) > 1 and pattern in gender_lower) 
                       for pattern in normalization_patterns['female_patterns']):
                    return 'Female'
                
                if any(pattern == gender_lower or (len(pattern) > 1 and pattern in gender_lower) 
                       for pattern in normalization_patterns['male_patterns']):
                    return 'Male'
                
                if any(pattern in gender_lower for pattern in normalization_patterns['nonbinary_patterns']):
                    return 'Non-binary'
                
                if any(pattern in gender_lower for pattern in normalization_patterns['na_patterns']):
                    return 'NA'
                
                return 'NA'
            
            # Create mapping
            gender_mapping = {}
            for gender_val in gender_values:
                normalized = normalize_gender_value(gender_val)
                gender_mapping[gender_val] = normalized
            
            print(f"Created {len(gender_mapping)} gender mappings")
            if len(gender_mapping) == 0:
                print("❌ No gender mappings created. Check normalization logic.")
                return
            
            # Create DataFrame
            mapping_df = pd.DataFrame([
                {'Original_Gender': k, 'Normalized_Gender': v}
                for k, v in gender_mapping.items()
            ])
            
            print(f"DataFrame created with columns: {list(mapping_df.columns)}")
            print(f"DataFrame shape: {mapping_df.shape}")
            
            # Save to CSV
            output_file = self.config['output_files']['gender']
            mapping_df.to_csv(output_file, index=False)
            
            print(f"✅ Gender mapping saved to: {output_file}")
            print(f"   - {len(mapping_df)} unique gender values processed")
            
            # Show distribution
            if len(mapping_df) > 0 and 'Normalized_Gender' in mapping_df.columns:
                distribution = mapping_df['Normalized_Gender'].value_counts()
                for category, count in distribution.items():
                    print(f"   - {category}: {count} values")
                self.stats['gender']['distribution'] = distribution.to_dict()
            else:
                print("❌ DataFrame is empty or missing columns, cannot show distribution")
                self.stats['gender']['distribution'] = {}
            
            self.stats['gender']['mapping_file'] = output_file
            self.stats['gender']['total_values'] = len(mapping_df)
            
        finally:
            conn.close()
    
    def apply_gender_mapping(self, fill_empty_with_na=False, informant_ids=None):
        """Apply gender mapping to database.
        
        Args:
            fill_empty_with_na: If True, fill empty mappings with "NA"
            informant_ids: If provided, only normalize these InformantIDs
        """
        # Load mapping file (check for manual override first)
        manual_file = self.config['manual_override_files']['gender']
        auto_file = self.config['output_files']['gender']
        
        if os.path.exists(manual_file):
            mapping_file = manual_file
            print(f"Using manual override file: {manual_file}")
        elif os.path.exists(auto_file):
            mapping_file = auto_file
            print(f"Using automatic mapping file: {auto_file}")
        else:
            print(f"❌ No mapping file found. Run update mode first.")
            return
        
        # Load mapping
        mapping_df = pd.read_csv(mapping_file, keep_default_na=False)
        
        # Fill empty mappings with NA if requested
        if fill_empty_with_na:
            empty_mask = (mapping_df['Normalized_Gender'].isna() | 
                         (mapping_df['Normalized_Gender'].astype(str).str.strip() == ''))
            empty_count = empty_mask.sum()
            if empty_count > 0:
                mapping_df.loc[empty_mask, 'Normalized_Gender'] = 'NA'
                print(f"⚠️  Filled {empty_count} empty gender mappings with 'NA'")
        
        mapping_dict = dict(zip(mapping_df['Original_Gender'], mapping_df['Normalized_Gender']))
        
        conn = sqlite3.connect(self.db_path)
        
        try:
            cursor = conn.cursor()
            
            # Add normalized column
            try:
                cursor.execute(f"ALTER TABLE {self.table_name} ADD COLUMN gender_normalized TEXT")
                print("Added 'gender_normalized' column")
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e).lower():
                    raise e
            
            # Apply mappings
            updated_count = 0
            id_filter = self._setup_id_filter(conn, informant_ids)
            for orig_gender, norm_gender in mapping_dict.items():
                db_value = 'NULL' if norm_gender == 'NA' else f"'{norm_gender}'"
                
                if orig_gender is None:
                    cursor.execute(f"""
                        UPDATE {self.table_name} 
                        SET gender_normalized = {db_value}
                        WHERE {self.config['columns']['gender']} IS NULL{id_filter}
                    """)
                else:
                    escaped_gender = str(orig_gender).replace("'", "''")
                    cursor.execute(f"""
                        UPDATE {self.table_name} 
                        SET gender_normalized = {db_value}
                        WHERE {self.config['columns']['gender']} = '{escaped_gender}'{id_filter}
                    """)
                updated_count += cursor.rowcount
            
            if informant_ids is not None:
                self._cleanup_id_filter(conn)
                updated_count += cursor.rowcount
            
            conn.commit()
            print(f"✅ Updated {updated_count} records with normalized gender values")
            
            # Verification
            verification_df = pd.read_sql_query(f'''
                SELECT 
                    CASE WHEN gender_normalized IS NULL THEN 'NULL' ELSE gender_normalized END as gender_normalized, 
                    COUNT(*) as Count
                FROM {self.table_name} 
                GROUP BY gender_normalized
                ORDER BY Count DESC
            ''', conn)
            
            print("Final gender distribution:")
            print(verification_df.to_string(index=False))
            
        finally:
            conn.close()
    
    # ===== SCHOOL NORMALIZATION =====
    
    def update_school_mappings(self):
        """Generate school mapping CSV files for primary and secondary schools."""
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Get unique school values
            primary_df = pd.read_sql_query(f'''
                SELECT DISTINCT {self.config['columns']['primary_school']} 
                FROM {self.table_name} 
                ORDER BY {self.config['columns']['primary_school']}
            ''', conn)
            
            secondary_df = pd.read_sql_query(f'''
                SELECT DISTINCT {self.config['columns']['secondary_school']} 
                FROM {self.table_name} 
                ORDER BY {self.config['columns']['secondary_school']}
            ''', conn)
            
            # School normalization patterns
            normalization_patterns = {
                'mixed_patterns': [
                    'voluntary aided', 'voluntary controlled', 'mixed', 'private, state',
                    'state, church', 'state and church', 'state + church', 'state/church',
                    'state, private', 'state and private', 'state; private', 'state/private',
                    'public, private', 'public and private', 'public; private', 'public/private',
                    'state-funded, private', 'state-funded and private'
                ],
                'state_patterns': [
                    'state', 'public', 'state-funded', 'government', 'comprehensive', 
                    'maintained', 'community', 'foundation', 'academy', 'local authority',
                    'governmental', 'grammar', 'central school'
                ],
                'private_patterns': [
                    'private', 'independent', 'fee-paying', 'preparatory', 'prep', 
                    'public school', 'boarding', 'semi-private', 'montessori', 'waldorf', 'steiner'
                ],
                'church_patterns': [
                    'church', 'religious', 'catholic', 'christian', 'rc', 
                    'faith', 'denominational', 'methodist', 'baptist', 'anglican',
                    'convent', 'episcopalian'
                ],
                'other_patterns': [
                    'special', 'tutorial', 'college', 'sixth form', 'further education',
                    'technical', 'vocational', 'adult', 'alternative', 'international',
                    'gymnasium', 'forces', 'army', 'orphan', 'municipality'
                ],
                'na_patterns': [
                    'unknown', 'not specified', 'not applicable', 'n/a', 'na', 'null', 
                    'missing', 'unspecified', '?', '-', 'none', 'other', '',
                    'dropped out', 'drop-out', 'war'
                ]
            }
            
            def normalize_school_value(school_val):
                if pd.isna(school_val) or school_val == '' or school_val is None:
                    return 'NA'
                
                school_lower = str(school_val).lower().strip()
                
                # Priority order: Mixed -> State -> Private -> Church -> Other -> NA
                for pattern in normalization_patterns['mixed_patterns']:
                    if pattern.lower() in school_lower:
                        return 'Mixed'
                
                for pattern in normalization_patterns['state_patterns']:
                    if pattern.lower() in school_lower:
                        return 'State'
                
                for pattern in normalization_patterns['private_patterns']:
                    if pattern.lower() in school_lower:
                        return 'Private'
                
                for pattern in normalization_patterns['church_patterns']:
                    if pattern.lower() in school_lower:
                        return 'Church'
                
                for pattern in normalization_patterns['other_patterns']:
                    if pattern.lower() in school_lower:
                        return 'Other'
                
                for pattern in normalization_patterns['na_patterns']:
                    if pattern.lower() in school_lower:
                        return 'NA'
                
                return 'NA'
            
            # Process primary schools
            primary_values = primary_df[self.config['columns']['primary_school']].tolist()
            print(f"Found {len(primary_values)} unique primary school values")
            
            if len(primary_values) == 0:
                print("❌ No primary school values found in database")
                primary_mapping_df = pd.DataFrame(columns=['Original_School', 'Normalized_School', 'School_Type'])
            else:
                primary_mapping = {val: normalize_school_value(val) for val in primary_values}
                primary_mapping_df = pd.DataFrame([
                    {'Original_School': k, 'Normalized_School': v, 'School_Type': 'Primary'}
                    for k, v in primary_mapping.items()
                ])
            
            # Process secondary schools
            secondary_values = secondary_df[self.config['columns']['secondary_school']].tolist()
            print(f"Found {len(secondary_values)} unique secondary school values")
            
            if len(secondary_values) == 0:
                print("❌ No secondary school values found in database")
                secondary_mapping_df = pd.DataFrame(columns=['Original_School', 'Normalized_School', 'School_Type'])
            else:
                secondary_mapping = {val: normalize_school_value(val) for val in secondary_values}
                secondary_mapping_df = pd.DataFrame([
                    {'Original_School': k, 'Normalized_School': v, 'School_Type': 'Secondary'}
                    for k, v in secondary_mapping.items()
                ])
            
            # Save mappings
            primary_file = self.config['output_files']['primary_school']
            secondary_file = self.config['output_files']['secondary_school']
            
            primary_mapping_df.to_csv(primary_file, index=False)
            secondary_mapping_df.to_csv(secondary_file, index=False)
            
            print(f"✅ Primary school mapping saved to: {primary_file}")
            print(f"   - {len(primary_mapping_df)} unique primary school values processed")
            
            print(f"✅ Secondary school mapping saved to: {secondary_file}")
            print(f"   - {len(secondary_mapping_df)} unique secondary school values processed")
            
            # Show distributions
            for school_type, mapping_df in [('Primary', primary_mapping_df), ('Secondary', secondary_mapping_df)]:
                if len(mapping_df) > 0 and 'Normalized_School' in mapping_df.columns:
                    distribution = mapping_df['Normalized_School'].value_counts()
                    print(f"   {school_type} school distribution:")
                    for category, count in distribution.items():
                        print(f"     - {category}: {count} values")
                else:
                    print(f"   {school_type} school distribution: No data available")
            
        finally:
            conn.close()
    
    def apply_school_mappings(self, fill_empty_with_na=False, informant_ids=None):
        """Apply school mappings to database.
        
        Args:
            fill_empty_with_na: If True, fill empty mappings with "NA"
            informant_ids: If provided, only normalize these InformantIDs
        """
        # Load mapping files
        for school_type in ['primary_school', 'secondary_school']:
            manual_file = self.config['manual_override_files'][school_type]
            auto_file = self.config['output_files'][school_type]
            
            if os.path.exists(manual_file):
                mapping_file = manual_file
                print(f"Using manual {school_type} file: {manual_file}")
            elif os.path.exists(auto_file):
                mapping_file = auto_file
                print(f"Using automatic {school_type} file: {auto_file}")
            else:
                print(f"❌ No {school_type} mapping file found. Run update mode first.")
                continue
            
            # Load and apply mapping
            mapping_df = pd.read_csv(mapping_file, keep_default_na=False)
            
            # Fill empty mappings with NA if requested
            if fill_empty_with_na:
                empty_mask = (mapping_df['Normalized_School'].isna() | 
                             (mapping_df['Normalized_School'].astype(str).str.strip() == ''))
                empty_count = empty_mask.sum()
                if empty_count > 0:
                    mapping_df.loc[empty_mask, 'Normalized_School'] = 'NA'
                    print(f"⚠️  Filled {empty_count} empty {school_type} mappings with 'NA'")
            
            mapping_dict = dict(zip(mapping_df['Original_School'], mapping_df['Normalized_School']))
            
            conn = sqlite3.connect(self.db_path)
            
            try:
                cursor = conn.cursor()
                
                # Determine column names
                original_column = self.config['columns'][school_type]
                normalized_column = f"{school_type}_normalized"
                
                # Add normalized column
                try:
                    cursor.execute(f"ALTER TABLE {self.table_name} ADD COLUMN {normalized_column} TEXT")
                    print(f"Added '{normalized_column}' column")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" not in str(e).lower():
                        raise e
                
                # Apply mappings
                updated_count = 0
                id_filter = self._setup_id_filter(conn, informant_ids)
                for orig_school, norm_school in mapping_dict.items():
                    db_value = 'NULL' if norm_school == 'NA' else f"'{norm_school}'"
                    
                    if orig_school is None:
                        cursor.execute(f"""
                            UPDATE {self.table_name} 
                            SET {normalized_column} = {db_value}
                            WHERE {original_column} IS NULL{id_filter}
                        """)
                    else:
                        escaped_school = str(orig_school).replace("'", "''")
                        cursor.execute(f"""
                            UPDATE {self.table_name} 
                            SET {normalized_column} = {db_value}
                            WHERE {original_column} = '{escaped_school}'{id_filter}
                        """)
                    updated_count += cursor.rowcount
                
                if informant_ids is not None:
                    self._cleanup_id_filter(conn)
                    updated_count += cursor.rowcount
                
                conn.commit()
                print(f"✅ Updated {updated_count} {school_type} records")
                
            finally:
                conn.close()
    
    # ===== QUALIFICATION MAPPING =====
    
    def update_qualification_mapping(self):
        """Generate qualification mapping CSV file."""
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Get unique qualifications from all qualification columns
            all_qualifications = set()
            qualification_columns = self.config['columns']['qualifications']
            
            # Ensure it's a list
            if isinstance(qualification_columns, str):
                qualification_columns = [qualification_columns]
            
            print(f"Processing qualification columns: {', '.join(qualification_columns)}")
            
            for column in qualification_columns:
                # Check if column exists
                cursor = conn.cursor()
                cursor.execute(f"PRAGMA table_info({self.table_name})")
                columns_info = cursor.fetchall()
                column_names = [col[1] for col in columns_info]
                
                if column not in column_names:
                    print(f"Warning: Column {column} not found. Skipping...")
                    continue
                
                # Get unique qualifications from this column
                df_quals = pd.read_sql_query(f'''
                    SELECT DISTINCT {column} 
                    FROM {self.table_name} 
                    WHERE {column} IS NOT NULL 
                    AND {column} != ""
                    ORDER BY {column}
                ''', conn)
                
                column_qualifications = set(df_quals[column].tolist())
                all_qualifications.update(column_qualifications)
                print(f"  - {column}: {len(column_qualifications)} unique values")
            
            qualifications = sorted(list(all_qualifications))
            
            print(f"Total unique qualification values: {len(qualifications)}")
            if len(qualifications) == 0:
                print("❌ No qualification values found in database. Check database path and column names.")
                return
            
            # Define qualification patterns
            pattern_lists = {
                'higher_education_patterns': [
                    'master', 'phd', 'doctorate', 'doctoral', 'postgraduate', 'post graduate',
                    'pgce', 'post-graduate', 'mbchb', 'medical degree', 'law degree', 'medicine', 
                    'bachelor', 'undergraduate', 'degree', 'university', 'bsc', 'ba ', 'llb'
                ],
                'professional_vocational_patterns': [
                    'professional', 'acca', 'chartered', 'licensed', 'qualified', 'certification',
                    'registered', 'vocational', 'nvq', 'btec', 'hnc', 'hnd', 'certificate',
                    'city and guilds', 'technical', 'college', 'tafe', 'associate degree'
                ],
                'work_based_patterns': [
                    'apprentice', 'work training', 'work-training', 'on-the-job', 'training'
                ],
                'secondary_patterns': [
                    'gcse', 'o-level', 'o-levels', 'a-level', 'a-levels', 'high school', 
                    'secondary', 'higher', 'sixth form', 'standard grade', 'advanced higher', 
                    'matriculation', 'leaving cert', 'abitur', 'highest level', 'form', 
                    'grade', 'cse', 'gce', 'hsc', 'diploma', 'level', 'levels'
                ],
                'no_other_patterns': [
                    'none', 'dropped', 'discontinued', '/', '?', 'pupil', 'housewife'
                ]
            }
            
            def categorize_qualification(qual):
                qual_lower = qual.lower()
                
                if any(pattern in qual_lower for pattern in pattern_lists['higher_education_patterns']):
                    return 'Higher Education'
                
                if any(pattern in qual_lower for pattern in pattern_lists['professional_vocational_patterns']):
                    return 'Professional/Vocational'
                
                if any(pattern in qual_lower for pattern in pattern_lists['work_based_patterns']):
                    return 'Work-Based Learning'
                
                if any(pattern in qual_lower for pattern in pattern_lists['secondary_patterns']):
                    return 'Secondary Education'
                
                if any(pattern in qual_lower for pattern in pattern_lists['no_other_patterns']):
                    return 'No/Other Qualification'
                
                return 'No/Other Qualification'
            
            # Create mapping
            mapping_dict = {}
            for qual in qualifications:
                category = categorize_qualification(qual)
                mapping_dict[qual] = category
            
            print(f"Created {len(mapping_dict)} qualification mappings")
            if len(mapping_dict) == 0:
                print("❌ No qualification mappings created. Check normalization logic.")
                return
            
            # Create DataFrame
            mapping_df = pd.DataFrame([
                {'Original_Qualification': k, 'Mapped_Category': v}
                for k, v in mapping_dict.items()
            ])
            
            # Save to CSV
            output_file = self.config['output_files']['qualifications']
            mapping_df.to_csv(output_file, index=False)
            
            print(f"✅ Qualification mapping saved to: {output_file}")
            print(f"   - {len(mapping_df)} unique qualification values processed")
            
            # Show distribution
            if len(mapping_df) > 0 and 'Mapped_Category' in mapping_df.columns:
                distribution = mapping_df['Mapped_Category'].value_counts()
                for category, count in distribution.items():
                    print(f"   - {category}: {count} values")
            else:
                print("❌ DataFrame is empty or missing columns, cannot show distribution")
            
        finally:
            conn.close()
    
    def apply_qualification_mapping(self, fill_empty_with_na=False, informant_ids=None):
        """Apply qualification mapping to database.
        
        Args:
            fill_empty_with_na: If True, fill empty mappings with "NA"
            informant_ids: If provided, only normalize these InformantIDs
        """
        # Load mapping file
        manual_file = self.config['manual_override_files']['qualifications']
        auto_file = self.config['output_files']['qualifications']
        
        if os.path.exists(manual_file):
            mapping_file = manual_file
            print(f"Using manual qualification file: {manual_file}")
        elif os.path.exists(auto_file):
            mapping_file = auto_file
            print(f"Using automatic qualification file: {auto_file}")
        else:
            print(f"❌ No qualification mapping file found. Run update mode first.")
            return
        
        # Load mapping
        mapping_df = pd.read_csv(mapping_file, keep_default_na=False)
        
        # Fill empty mappings with NA if requested
        if fill_empty_with_na:
            empty_mask = (mapping_df['Mapped_Category'].isna() | 
                         (mapping_df['Mapped_Category'].astype(str).str.strip() == ''))
            empty_count = empty_mask.sum()
            if empty_count > 0:
                mapping_df.loc[empty_mask, 'Mapped_Category'] = 'NA'
                print(f"⚠️  Filled {empty_count} empty qualification mappings with 'NA'")
        
        mapping_dict = dict(zip(mapping_df['Original_Qualification'], mapping_df['Mapped_Category']))
        
        conn = sqlite3.connect(self.db_path)
        
        try:
            cursor = conn.cursor()
            
            # Get qualification columns
            qualification_columns = self.config['columns']['qualifications']
            if isinstance(qualification_columns, str):
                qualification_columns = [qualification_columns]
            
            print(f"Applying mappings to columns: {', '.join(qualification_columns)}")
            
            # Process each qualification column
            for column in qualification_columns:
                # Check if column exists
                cursor.execute(f"PRAGMA table_info({self.table_name})")
                columns_info = cursor.fetchall()
                column_names = [col[1] for col in columns_info]
                
                if column not in column_names:
                    print(f"Warning: Column {column} not found. Skipping...")
                    continue
                
                # Create normalized column name
                normalized_column = f"{column}_normalized"
                
                # Add normalized column
                try:
                    cursor.execute(f"ALTER TABLE {self.table_name} ADD COLUMN {normalized_column} TEXT")
                    print(f"Added '{normalized_column}' column")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" not in str(e).lower():
                        raise e
                
                # Apply mappings
                updated_count = 0
                id_filter = self._setup_id_filter(conn, informant_ids)
                for qual, category in mapping_dict.items():
                    escaped_qual = qual.replace("'", "''")
                    cursor.execute(f"""
                        UPDATE {self.table_name} 
                        SET {normalized_column} = '{category}' 
                        WHERE {column} = '{escaped_qual}'{id_filter}
                    """)
                    updated_count += cursor.rowcount
                if informant_ids is not None:
                    self._cleanup_id_filter(conn)
                    updated_count += cursor.rowcount
                
                print(f"✅ Updated {updated_count} records in {column} → {normalized_column}")
            
            # Special handling for main 'Qualifications' column (legacy support)
            # Also create 'highest_qualification' as an alias for Qualifications_normalized
            if 'Qualifications' in qualification_columns:
                try:
                    cursor.execute(f"ALTER TABLE {self.table_name} ADD COLUMN highest_qualification TEXT")
                    print("Added 'highest_qualification' column (legacy alias)")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" not in str(e).lower():
                        raise e
                
                # Copy from Qualifications_normalized to highest_qualification
                id_filter = self._setup_id_filter(conn, informant_ids)
                cursor.execute(f"""
                    UPDATE {self.table_name}
                    SET highest_qualification = Qualifications_normalized
                    WHERE Qualifications_normalized IS NOT NULL{id_filter}
                """)
                if informant_ids is not None:
                    self._cleanup_id_filter(conn)
            
            # Apply InformantID-specific mappings if available
            id_mapping_file = self.config['manual_override_files']['qualifications_by_id']
            if os.path.exists(id_mapping_file):
                print(f"Applying InformantID-specific mappings from: {id_mapping_file}")
                id_mapping_df = pd.read_csv(id_mapping_file, keep_default_na=False)
                
                if 'InformantID' in id_mapping_df.columns and 'Mapped_Category' in id_mapping_df.columns:
                    # If filtering by IDs, only apply overrides for new participants
                    if informant_ids is not None:
                        id_mapping_df = id_mapping_df[id_mapping_df['InformantID'].isin(informant_ids)]
                    
                    id_updates = 0
                    for _, row in id_mapping_df.iterrows():
                        # Apply to highest_qualification (main column)
                        cursor.execute("""
                            UPDATE Informants 
                            SET highest_qualification = ? 
                            WHERE InformantID = ?
                        """, (row['Mapped_Category'], row['InformantID']))
                        
                        # Also update Qualifications_normalized if it exists
                        if 'Qualifications' in qualification_columns:
                            cursor.execute("""
                                UPDATE Informants 
                                SET Qualifications_normalized = ? 
                                WHERE InformantID = ?
                            """, (row['Mapped_Category'], row['InformantID']))
                        
                        id_updates += cursor.rowcount
                    print(f"Applied {len(id_mapping_df)} InformantID-specific overrides")
            
            # Update qualifications based on secondary school data (for main Qualifications column only)
            if 'Qualifications' in qualification_columns:
                id_filter = self._setup_id_filter(conn, informant_ids)
                cursor.execute(f"""
                    UPDATE {self.table_name} 
                    SET highest_qualification = 'Secondary Education'
                    WHERE highest_qualification IS NULL 
                    AND secondary_school_normalized IS NOT NULL
                    AND secondary_school_normalized != ''{id_filter}
                """)
                secondary_updates = cursor.rowcount
                
                if secondary_updates > 0:
                    print(f"✅ Inferred {secondary_updates} 'Secondary Education' qualifications from school data")
            
            conn.commit()
            print(f"✅ All qualification mappings applied successfully")
            
        finally:
            conn.close()
    
    # ===== LANGUAGE NORMALIZATION =====
    
    def update_language_mapping(self):
        """Generate language mapping CSV file."""
        print("Attempting to load Flair NER model for language detection...")
        
        # Try to load Flair, but continue without it if unavailable
        try:
            from flair.data import Sentence
            from flair.models import SequenceTagger
            tagger = SequenceTagger.load("flair/ner-english-fast")
            use_ner = True
            print("✅ Flair NER model loaded successfully")
        except Exception as e:
            print(f"⚠️  Warning: Could not load Flair model: {e}")
            print("Continuing with regex-only normalization...")
            use_ner = False
        
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Get all unique language values from all language columns
            all_raw_values = set()
            
            for column in self.config['columns']['languages']:
                # Check if column exists
                cursor = conn.cursor()
                cursor.execute(f"PRAGMA table_info({self.table_name})")
                columns_info = cursor.fetchall()
                column_names = [col[1] for col in columns_info]
                
                if column not in column_names:
                    print(f"Warning: Column {column} not found. Skipping...")
                    continue
                
                # Extract unique values
                query = f"""
                    SELECT DISTINCT {column}
                    FROM {self.table_name}
                    WHERE {column} IS NOT NULL 
                    AND {column} != ''
                    AND TRIM({column}) != ''
                """
                
                column_df = pd.read_sql_query(query, conn)
                column_values = set(column_df[column].astype(str))
                all_raw_values.update(column_values)
            
            print(f"Processing {len(all_raw_values)} unique language values...")
            
            if len(all_raw_values) == 0:
                print("❌ No language values found in database. Check database path and column names.")
                return
            
            # Apply NER and regex normalization
            mapping_data = []
            processed_count = 0
            
            for raw_value in sorted(all_raw_values):
                processed_count += 1
                if processed_count % 50 == 0:
                    print(f"  Processed {processed_count}/{len(all_raw_values)} values...")
                
                # Apply NER if available
                ner_result = ""
                if use_ner:
                    try:
                        sentence = Sentence(str(raw_value))
                        tagger.predict(sentence)
                        entities = [entity.text for entity in sentence.get_spans('ner')]
                        ner_result = ", ".join(entities) if entities else ""
                    except Exception as e:
                        print(f"Warning: NER failed for '{raw_value}': {e}")
                
                # Apply regex normalization
                normalized_result = self._normalize_language_with_regex(ner_result if ner_result else raw_value)
                
                mapping_data.append({
                    'Original_Language': raw_value,
                    'NER_Result': ner_result,
                    'Normalized_Language': normalized_result
                })
            
            # Create DataFrame
            mapping_df = pd.DataFrame(mapping_data)
            
            # Save to CSV
            output_file = self.config['output_files']['languages']
            mapping_df.to_csv(output_file, index=False)
            
            print(f"✅ Language mapping saved to: {output_file}")
            print(f"   - {len(mapping_df)} unique language values processed")
            
            if use_ner:
                ner_success = len(mapping_df[mapping_df['NER_Result'] != ''])
                print(f"   - NER found languages in: {ner_success} values ({ner_success/len(mapping_df)*100:.1f}%)")
            
            regex_success = len(mapping_df[mapping_df['Normalized_Language'] != ''])
            print(f"   - Final normalization successful: {regex_success} values ({regex_success/len(mapping_df)*100:.1f}%)")
            
        finally:
            conn.close()
    
    def _normalize_language_with_regex(self, text):
        """Apply regex-based language normalization."""
        if pd.isna(text) or text == '':
            return ''
        
        # Split by common separators and clean
        languages = re.split(r'[,;/&+]|(?:\s+and\s+)', str(text).lower())
        languages = [lang.strip() for lang in languages if lang.strip()]
        
        # Normalization patterns
        normalization_patterns = {
            r'\b(?:british\s+|american\s+|canadian\s+|australian\s+|indian\s+|south\s+african\s+)?english\b': 'English',
            r'\b(?:castilian\s+|latin\s+american\s+|mexican\s+)?spanish\b': 'Spanish',
            r'\bespañol\b': 'Spanish',
            r'\b(?:canadian\s+|belgian\s+|swiss\s+)?french\b': 'French',
            r'\bfrançais\b': 'French',
            r'\b(?:high\s+|standard\s+|austrian\s+|swiss\s+)?german\b': 'German',
            r'\bdeutsch\b': 'German',
            r'\b(?:brazilian\s+|european\s+)?portuguese\b': 'Portuguese',
            r'\bitalian\b': 'Italian',
            r'\bdutch\b': 'Dutch',
            r'\bflemish\b': 'Dutch',
            r'\barabic\b': 'Arabic',
            r'\bmandarin(?:\s+chinese)?\b': 'Mandarin',
            r'\bcantonese(?:\s+chinese)?\b': 'Cantonese',
            r'\brussian\b': 'Russian',
            r'\bjapanese\b': 'Japanese',
            r'\bkorean\b': 'Korean',
            r'\bhindi\b': 'Hindi',
            r'\bwelsh\b': 'Welsh',
            r'\bbritish\s+sign\s+language\b': 'British Sign Language',
            r'\bbsl\b': 'British Sign Language',
        }
        
        normalized_languages = []
        for lang in languages:
            lang = lang.strip()
            if not lang:
                continue
            
            # Apply normalization patterns
            normalized = lang
            for pattern, replacement in normalization_patterns.items():
                normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
            
            # Clean up and title case
            normalized = ' '.join(normalized.split()).title()
            
            if normalized and len(normalized) > 1:
                normalized_languages.append(normalized)
        
        # Remove duplicates while preserving order
        seen = set()
        final_languages = []
        for lang in normalized_languages:
            lang_lower = lang.lower()
            if lang_lower not in seen:
                seen.add(lang_lower)
                final_languages.append(lang)
        
        return ', '.join(final_languages)
    
    def apply_language_mapping(self, fill_empty_with_na=False, informant_ids=None):
        """Apply language mapping to database.
        
        Args:
            fill_empty_with_na: If True, fill empty mappings with "NA"
            informant_ids: If provided, only normalize these InformantIDs
        """
        # Load mapping file
        manual_file = self.config['manual_override_files']['languages']
        auto_file = self.config['output_files']['languages']
        
        if os.path.exists(manual_file):
            mapping_file = manual_file
            print(f"Using manual language file: {manual_file}")
        elif os.path.exists(auto_file):
            mapping_file = auto_file
            print(f"Using automatic language file: {auto_file}")
        else:
            print(f"❌ No language mapping file found. Run update mode first.")
            return
        
        # Load mapping
        mapping_df = pd.read_csv(mapping_file, keep_default_na=False)
        
        # Fill empty mappings with NA if requested
        if fill_empty_with_na:
            empty_mask = (mapping_df['Normalized_Language'].isna() | 
                         (mapping_df['Normalized_Language'].astype(str).str.strip() == ''))
            empty_count = empty_mask.sum()
            if empty_count > 0:
                mapping_df.loc[empty_mask, 'Normalized_Language'] = 'NA'
                print(f"⚠️  Filled {empty_count} empty language mappings with 'NA'")
        
        mapping_dict = dict(zip(mapping_df['Original_Language'], mapping_df['Normalized_Language']))
        
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Determine which rows to process
            if informant_ids is not None:
                placeholders = ",".join(["?"] * len(informant_ids))
                df = pd.read_sql_query(
                    f"SELECT * FROM {self.table_name} WHERE InformantID IN ({placeholders})",
                    conn, params=list(informant_ids),
                )
            else:
                df = pd.read_sql_query(f"SELECT * FROM {self.table_name}", conn)
            
            # Process each language column
            for column in self.config['columns']['languages']:
                if column not in df.columns:
                    print(f"Warning: Column {column} not found. Skipping...")
                    continue
                
                normalized_column = f"{column}_normalized"
                
                # Ensure normalized column exists in DB
                cursor = conn.cursor()
                try:
                    cursor.execute(f"ALTER TABLE {self.table_name} ADD COLUMN {normalized_column} TEXT")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" not in str(e).lower():
                        raise e
                
                # Apply mapping
                def apply_mapping(value):
                    if pd.isna(value) or value == '':
                        return ''
                    
                    str_value = str(value)
                    if str_value in mapping_dict:
                        return mapping_dict[str_value]
                    else:
                        return ''
                
                df[normalized_column] = df[column].apply(apply_mapping)
                
                # Calculate statistics
                total_values = len(df[df[column].notna() & (df[column] != '')])
                mapped_values = len(df[df[normalized_column] != ''])
                mapping_rate = (mapped_values / total_values * 100) if total_values > 0 else 0
                
                print(f"✅ {column}: {mapped_values}/{total_values} values mapped ({mapping_rate:.1f}%)")
                
                # Write back using row-level UPDATEs (safe for both full and incremental)
                for _, row in df.iterrows():
                    norm_val = row.get(normalized_column, '')
                    cursor.execute(
                        f"UPDATE {self.table_name} SET {normalized_column} = ? WHERE InformantID = ?",
                        (norm_val, row['InformantID']),
                    )
            
            conn.commit()
            print(f"✅ Language mappings applied to database")
            
        finally:
            conn.close()
    
    # ===== SUMMARY METHODS =====
    
    def print_update_summary(self):
        """Print summary after update mode."""
        print("🎯 UPDATE MODE SUMMARY")
        print("=" * 80)
        print("Generated/updated the following CSV mapping files:")
        print()
        
        for category, filename in self.config['output_files'].items():
            if os.path.exists(filename):
                print(f"✅ {category.replace('_', ' ').title()}: {filename}")
            else:
                print(f"❌ {category.replace('_', ' ').title()}: {filename} (failed)")
        
        print()
        print("📝 MANUAL MAPPING FILES:")
        print()
        
        for category, manual_file in self.config['manual_override_files'].items():
            if os.path.exists(manual_file):
                # Show file size/entries
                try:
                    manual_df = pd.read_csv(manual_file, keep_default_na=False)
                    print(f"✅ {manual_file} ({len(manual_df)} entries)")
                except:
                    print(f"✅ {manual_file}")
            else:
                print(f"⚠️  {manual_file} (not created yet)")
        
        print()
        print("📝 NEXT STEPS:")
        print("1. Review the manual mapping files in mappings/ folder")
        print("2. New values have been automatically added with auto-generated mappings")
        print("3. Edit the mappings for any values that need correction")
        print("4. Run in apply mode to update the database:")
        print("   python unified_data_normalization.py --mode apply")
        print()
        print("💡 TIP: Manual files are automatically synced with new values from the database")
        print("   You don't need to manually copy files anymore!")
    
    def print_apply_summary(self):
        """Print summary after apply mode."""
        print("🎯 APPLY MODE SUMMARY")
        print("=" * 80)
        print("Applied mappings to the following database columns:")
        print()
        print("✅ gender_normalized")
        print("✅ primary_school_normalized")
        print("✅ secondary_school_normalized")
        
        # Show all qualification columns
        qualification_columns = self.config['columns']['qualifications']
        if isinstance(qualification_columns, str):
            qualification_columns = [qualification_columns]
        
        for qual_col in qualification_columns:
            print(f"✅ {qual_col}_normalized")
        
        if 'Qualifications' in qualification_columns:
            print("✅ highest_qualification (legacy alias for Qualifications_normalized)")
        
        for lang_col in self.config['columns']['languages']:
            print(f"✅ {lang_col}_normalized")
        
        print()
        print("📊 DATABASE UPDATED SUCCESSFULLY")
        print("The normalized columns are now available for analysis.")
        print()
        print("📁 All mapping files are organized in the 'mappings/' folder")
        print("💡 TIP: To make further changes:")
        print("1. Edit the manual override CSV files in the mappings/ folder")
        print("2. Re-run in apply mode to update the database")


def run_cleansing(mode="apply", fill_empty_with_na=False, informant_ids=None):
    """
    Public entry point for the data cleansing stage.

    Parameters
    ----------
    mode : "update" | "apply"
    fill_empty_with_na : bool
    informant_ids : list[str] | None
        If provided, only normalize these InformantIDs (for incremental updates).
    """
    print()
    print("=" * 80)
    print(f"  STAGE: DATA CLEANSING – mode={mode}")
    print("=" * 80)

    normalizer = UnifiedDataNormalizer()

    if mode == "update":
        normalizer.update_all_mappings()
    else:
        normalizer.apply_all_mappings(
            fill_empty_with_na=fill_empty_with_na,
            informant_ids=informant_ids,
        )

    print("  ✅ Data cleansing completed successfully")
