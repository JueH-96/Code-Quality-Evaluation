#!/usr/bin/env python3
"""
SonarQube Code-Level Correlation Analyzer (Fixed Version)
Analyzes Pearson correlations between SonarQube attributes across all codes under the same LLM
Maintains original column naming convention and ensures all LLMs are processed
"""

import os
import csv
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import glob
from datetime import datetime

# Check if openpyxl is available
try:
    import openpyxl
    from openpyxl.styles import Font, Alignment, PatternFill
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    print("Note: openpyxl not installed. Only CSV files will be generated. Install with: pip install openpyxl")


class SonarQubeCorrelationAnalyzer:
    def __init__(self, sonarqube_result_dir="sonarqube_result", output_dir="sonarqube_code_level_result"):
        """
        Initialize correlation analyzer
        
        Args:
            sonarqube_result_dir: SonarQube results directory
            output_dir: output directory
        """
        self.sonarqube_result_dir = sonarqube_result_dir
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Store analysis results
        self.correlation_results = []
        
        # Load total code counts from summary file
        self.total_code_counts = self._load_total_code_counts()
    
    def _load_total_code_counts(self) -> Dict[str, int]:
        """
        Load total code counts from sonarqube_summary.csv
        
        Returns:
            Dictionary mapping LLM names to total code counts
        """
        summary_file = os.path.join(self.sonarqube_result_dir, "sonarqube_summary.csv")
        
        if not os.path.exists(summary_file):
            print(f"Warning: {summary_file} not found. Will use actual data counts.")
            return {}
        
        try:
            df = pd.read_csv(summary_file)
            
            # Check required columns
            if 'LLM' not in df.columns or 'Total Codes' not in df.columns:
                print(f"Warning: Required columns not found in {summary_file}")
                return {}
            
            # Build mapping dictionary
            counts = {}
            for _, row in df.iterrows():
                llm_name = str(row['LLM']).strip()
                total_codes = int(row['Total Codes'])
                counts[llm_name] = total_codes
            
            print(f"\n✓ Loaded total code counts from {summary_file}")
            print(f"  Found {len(counts)} LLM entries")
            
            return counts
            
        except Exception as e:
            print(f"Warning: Error reading {summary_file}: {e}")
            return {}
    
    def _get_total_codes(self, llm_name: str, difficulty: str = None) -> int:
        """
        Get total code count for an LLM (with optional difficulty)
        
        Args:
            llm_name: LLM name
            difficulty: Optional difficulty level ('easy', 'medium', 'hard')
            
        Returns:
            Total code count
        """
        if difficulty:
            # For difficulty-specific queries, construct the key
            key = f"{llm_name} ({difficulty})"
        else:
            # For overall queries
            key = llm_name
        
        return self.total_code_counts.get(key, None)
    
    def read_llm_code_data(self, csv_file: str) -> pd.DataFrame:
        """
        Read LLM code detailed data
        All codes will be kept and counted without filtering
        
        Args:
            csv_file: CSV file path
            
        Returns:
            DataFrame containing code data
        """
        try:
            df = pd.read_csv(csv_file)
            
            # Check if CSV is empty
            if df.empty:
                print(f"  Warning: File {csv_file} is empty")
                return pd.DataFrame()
            
            original_count = len(df)
            print(f"  Read {original_count} codes from CSV file")
            
            # Convert Passed Tests to numeric format immediately after reading
            if 'Passed Tests' in df.columns:
                df = self._convert_passed_tests_to_numeric(df)
            
            # Return all data without filtering
            return df
            
        except FileNotFoundError:
            print(f"  Error: File not found {csv_file}")
            return pd.DataFrame()
        except Exception as e:
            print(f"  Error reading file {csv_file}: {e}")
            import traceback
            print(f"  Detailed error: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def _convert_passed_tests_to_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert Passed Tests column to numeric format
        
        Args:
            df: DataFrame with Passed Tests column
            
        Returns:
            DataFrame with numeric Passed Tests
        """
        def parse_passed_tests(value):
            """Convert Passed Tests to numeric (1 for pass, 0 for fail)"""
            if pd.isna(value):
                return np.nan
            
            value_str = str(value).strip().lower()
            
            # Format: "Yes"/"No" (primary format)
            if value_str in ['yes', 'y']:
                return 1.0
            elif value_str in ['no', 'n']:
                return 0.0
            
            # Format: "Pass"/"Fail"
            if value_str in ['pass', 'passed']:
                return 1.0
            elif value_str in ['fail', 'failed']:
                return 0.0
            
            # Format: "True"/"False" or 1/0
            if value_str in ['true', '1', '1.0']:
                return 1.0
            elif value_str in ['false', '0', '0.0']:
                return 0.0
            
            # Format: "3/5" -> calculate pass rate
            if '/' in value_str:
                try:
                    parts = value_str.split('/')
                    passed = float(parts[0])
                    total = float(parts[1])
                    return passed / total if total > 0 else 0.0
                except:
                    return np.nan
            
            # Format: "100%" -> convert to decimal
            if '%' in value_str:
                try:
                    return float(value_str.replace('%', '')) / 100.0
                except:
                    return np.nan
            
            # Try direct numeric conversion
            try:
                return float(value_str)
            except:
                return np.nan
        
        # Apply conversion
        df['Passed Tests'] = df['Passed Tests'].apply(parse_passed_tests)
        
        # Count conversion results
        valid_count = df['Passed Tests'].notna().sum()
        print(f"  Converted Passed Tests: {valid_count} valid values")
        
        return df
    
    def calculate_correlations(self, df: pd.DataFrame) -> Dict:
        """
        Calculate Pearson correlation coefficients between attributes across all code samples
        This calculates column-to-column correlations (attribute-to-attribute)
        
        Args:
            df: DataFrame containing code evaluation data (rows=codes, columns=attributes)
            
        Returns:
            Dictionary of correlation coefficients with original naming convention
        """
        if df.empty or len(df) < 3:
            # Too little data to calculate meaningful correlations
            return {}
        
        # Print all available columns for debugging
        print(f"    Available columns in CSV: {list(df.columns)}")
        
        # Select numeric columns for analysis - SonarQube metrics
        numeric_columns = [
            'Passed Tests',
            'Total Issues',
            # Severity metrics
            'Severity: Blocker',
            'Severity: Critical',
            'Severity: Major',
            'Severity: Minor',
            'Severity: Info',
            # Type metrics
            'Type: Bug',
            'Type: Vulnerability',
            'Type: Code Smell',
            'Type: Security Hotspot',
            # Clean Code Attributes
            'Attribute: Consistency',
            'Attribute: Intentionality',
            'Attribute: Adaptability',
            'Attribute: Responsibility',
            # Software Quality - Security
            'SQ Security: Blocker',
            'SQ Security: High',
            'SQ Security: Medium',
            'SQ Security: Low',
            'SQ Security: Info',
            # Software Quality - Reliability
            'SQ Reliability: Blocker',
            'SQ Reliability: High',
            'SQ Reliability: Medium',
            'SQ Reliability: Low',
            'SQ Reliability: Info',
            # Software Quality - Maintainability
            'SQ Maintainability: Blocker',
            'SQ Maintainability: High',
            'SQ Maintainability: Medium',
            'SQ Maintainability: Low',
            'SQ Maintainability: Info'
        ]
        
        # Ensure these columns exist
        available_columns = [col for col in numeric_columns if col in df.columns]
        
        print(f"    Columns used for correlation: {available_columns}")
        
        if not available_columns:
            return {}
        
        # Extract data
        data = df[available_columns]
        
        # Calculate correlation coefficient for each pair
        corr_results = {}
        
        # Define abbreviation mapping (for generating short keys)
        abbrev_map = {
            'Passed Tests': 'PT',
            'Total Issues': 'TOTAL',
            # Severity
            'Severity: Blocker': 'SEV_BLK',
            'Severity: Critical': 'SEV_CRT',
            'Severity: Major': 'SEV_MAJ',
            'Severity: Minor': 'SEV_MIN',
            'Severity: Info': 'SEV_INF',
            # Type
            'Type: Bug': 'TYP_BUG',
            'Type: Vulnerability': 'TYP_VUL',
            'Type: Code Smell': 'TYP_CS',
            'Type: Security Hotspot': 'TYP_SEC',
            # Attributes
            'Attribute: Consistency': 'ATR_CON',
            'Attribute: Intentionality': 'ATR_INT',
            'Attribute: Adaptability': 'ATR_ADP',
            'Attribute: Responsibility': 'ATR_RSP',
            # SQ Security
            'SQ Security: Blocker': 'SQS_BLK',
            'SQ Security: High': 'SQS_HI',
            'SQ Security: Medium': 'SQS_MED',
            'SQ Security: Low': 'SQS_LO',
            'SQ Security: Info': 'SQS_INF',
            # SQ Reliability
            'SQ Reliability: Blocker': 'SQR_BLK',
            'SQ Reliability: High': 'SQR_HI',
            'SQ Reliability: Medium': 'SQR_MED',
            'SQ Reliability: Low': 'SQR_LO',
            'SQ Reliability: Info': 'SQR_INF',
            # SQ Maintainability
            'SQ Maintainability: Blocker': 'SQM_BLK',
            'SQ Maintainability: High': 'SQM_HI',
            'SQ Maintainability: Medium': 'SQM_MED',
            'SQ Maintainability: Low': 'SQM_LO',
            'SQ Maintainability: Info': 'SQM_INF'
        }
        
        # Calculate Pearson correlation coefficient for each pair of attributes
        # This iterates through all pairs of columns (attributes)
        for i, col1 in enumerate(available_columns):
            for col2 in available_columns[i+1:]:
                try:
                    # Use all data (remove NaN values)
                    valid_data = data[[col1, col2]].dropna()
                    
                    if len(valid_data) < 3:
                        continue
                    
                    # Skip if either column has zero variance
                    if valid_data[col1].std() == 0 or valid_data[col2].std() == 0:
                        continue
                    
                    # Calculate Pearson correlation coefficient and p-value
                    # This correlates the VALUES of col1 with VALUES of col2 across all code samples
                    r, p_value = stats.pearsonr(valid_data[col1], valid_data[col2])
                    
                    # Use abbreviations as keys
                    key = f"{abbrev_map[col1]}-{abbrev_map[col2]}"
                    corr_results[key] = {
                        'r': r,
                        'p_value': p_value,
                        'n': len(valid_data)
                    }
                except Exception as e:
                    print(f"    Error calculating {col1} vs {col2} correlation: {e}")
                    continue
        
        return corr_results
    
    def format_correlation_for_csv(self, corr_dict: Dict) -> Dict:
        """
        Format correlation dictionary as CSV row with original naming convention
        
        Args:
            corr_dict: Correlation dictionary
            
        Returns:
            Formatted dictionary
        """
        formatted = {}
        
        if not corr_dict:
            return formatted
        
        # Extract number of codes
        n_codes = corr_dict.pop('n_codes', 0)
        total_codes = corr_dict.pop('total_codes', 'N/A')
        
        formatted['N_Codes'] = n_codes
        formatted['Total_Codes'] = total_codes
        
        # Define metric name mapping (from abbreviation to full name)
        metric_names = {
            'PT': 'Passed Tests',
            'TOTAL': 'Total Issues',
            # Severity
            'SEV_BLK': 'Severity Blocker',
            'SEV_CRT': 'Severity Critical',
            'SEV_MAJ': 'Severity Major',
            'SEV_MIN': 'Severity Minor',
            'SEV_INF': 'Severity Info',
            # Type
            'TYP_BUG': 'Type Bug',
            'TYP_VUL': 'Type Vulnerability',
            'TYP_CS': 'Type Code Smell',
            'TYP_SEC': 'Type Security Hotspot',
            # Attributes
            'ATR_CON': 'Attr Consistency',
            'ATR_INT': 'Attr Intentionality',
            'ATR_ADP': 'Attr Adaptability',
            'ATR_RSP': 'Attr Responsibility',
            # SQ Security
            'SQS_BLK': 'SQ Security Blocker',
            'SQS_HI': 'SQ Security High',
            'SQS_MED': 'SQ Security Medium',
            'SQS_LO': 'SQ Security Low',
            'SQS_INF': 'SQ Security Info',
            # SQ Reliability
            'SQR_BLK': 'SQ Reliability Blocker',
            'SQR_HI': 'SQ Reliability High',
            'SQR_MED': 'SQ Reliability Medium',
            'SQR_LO': 'SQ Reliability Low',
            'SQR_INF': 'SQ Reliability Info',
            # SQ Maintainability
            'SQM_BLK': 'SQ Maintainability Blocker',
            'SQM_HI': 'SQ Maintainability High',
            'SQM_MED': 'SQ Maintainability Medium',
            'SQM_LO': 'SQ Maintainability Low',
            'SQM_INF': 'SQ Maintainability Info'
        }
        
        # Iterate through each correlation pair
        for pair_name, values in corr_dict.items():
            if isinstance(values, dict) and 'r' in values:
                # Parse pair_name, e.g. "PT-TOTAL" -> "Passed Tests vs Total Issues"
                parts = pair_name.split('-')
                if len(parts) == 2:
                    metric1 = metric_names.get(parts[0], parts[0])
                    metric2 = metric_names.get(parts[1], parts[1])
                    
                    # Build full column name (original format)
                    base_name = f"{metric1} vs {metric2}"
                    
                    formatted[f"{base_name} (r)"] = round(values['r'], 4)
                    formatted[f"{base_name} (p-value)"] = round(values['p_value'], 4) if values['p_value'] >= 0.0001 else '<0.0001'
                    # Add sample size information
                    formatted[f"{base_name} (n)"] = values.get('n', n_codes)
        
        return formatted
    
    def analyze_llm(self, llm_name: str) -> Dict:
        """
        Analyze correlations for a single LLM (overall + by difficulty)
        
        Args:
            llm_name: LLM name
            
        Returns:
            Dictionary containing analysis results
        """
        print(f"Analyzing: {llm_name}")
        
        result = {
            'llm_name': llm_name,
            'all': None,
            'easy': None,
            'medium': None,
            'hard': None
        }
        
        # 1. Analyze overall data
        overall_file = os.path.join(self.sonarqube_result_dir, f"{llm_name}_Code_SonarQube.csv")
        if os.path.exists(overall_file):
            print(f"  Processing overall data...")
            df = self.read_llm_code_data(overall_file)
            
            if not df.empty:
                total_codes = self._get_total_codes(llm_name)
                correlations = self.calculate_correlations(df)
                correlations['n_codes'] = len(df)
                correlations['total_codes'] = total_codes if total_codes else len(df)
                result['all'] = correlations
                
                if total_codes and total_codes != len(df):
                    print(f"  ⚠ Note: CSV has {len(df)} codes, but summary shows {total_codes} total codes")
                    print(f"    Missing codes (no issues): {total_codes - len(df)}")
        
        # 2. Analyze data by difficulty
        difficulties = ['Easy', 'Medium', 'Hard']
        for difficulty in difficulties:
            difficulty_file = os.path.join(self.sonarqube_result_dir, 
                                         f"{llm_name}_{difficulty}_Code_SonarQube.csv")
            
            if os.path.exists(difficulty_file):
                print(f"  Processing {difficulty} difficulty...")
                df = self.read_llm_code_data(difficulty_file)
                
                if not df.empty:
                    total_codes = self._get_total_codes(llm_name, difficulty.lower())
                    correlations = self.calculate_correlations(df)
                    correlations['n_codes'] = len(df)
                    correlations['total_codes'] = total_codes if total_codes else len(df)
                    result[difficulty.lower()] = correlations
                    
                    if total_codes and total_codes != len(df):
                        print(f"  ⚠ Note: CSV has {len(df)} codes, but summary shows {total_codes} total codes")
                        print(f"    Missing codes (no issues): {total_codes - len(df)}")
        
        return result
    
    def interpret_correlation(self, r, p):
        """
        Interpret correlation coefficient and p-value in English
        
        Args:
            r: Pearson correlation coefficient
            p: p-value
            
        Returns:
            Interpretation text
        """
        if pd.isna(r) or pd.isna(p):
            return "No data"
        
        # Correlation strength
        abs_r = abs(r)
        if abs_r >= 0.7:
            strength = "Strong"
        elif abs_r >= 0.4:
            strength = "Moderate"
        else:
            strength = "Weak"
        
        # Correlation direction
        direction = "positive" if r > 0 else "negative"
        
        # Significance
        if p < 0.001:
            significance = "extremely significant"
        elif p < 0.01:
            significance = "highly significant"
        elif p < 0.05:
            significance = "significant"
        else:
            significance = "not significant"
        
        return f"{strength} {direction} correlation ({significance})"
    
    def generate_readable_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate readable correlation report DataFrame
        
        Args:
            df: Raw correlation data
            
        Returns:
            Readable report DataFrame
        """
        report_rows = []
        
        for _, row in df.iterrows():
            llm = row['LLM']
            difficulty = row['Difficulty']
            n_codes = row['N_Codes']
            total_codes = row.get('Total_Codes', 'N/A')
            
            # Extract all correlation pairs
            correlations = {}
            for col in row.index:
                if col.endswith(' (r)'):
                    pair_name = col[:-4]  # Remove ' (r)'
                    p_col = pair_name + ' (p-value)'
                    n_col = pair_name + ' (n)'
                    
                    if p_col in row.index:
                        r_value = row[col]
                        p_value = row[p_col]
                        n_value = row[n_col] if n_col in row.index else n_codes
                        
                        # Convert p-value (may be string "<0.0001")
                        if isinstance(p_value, str) and p_value.startswith('<'):
                            p_numeric = 0.0
                            p_str = p_value
                        elif pd.isna(p_value):
                            p_numeric = 1.0
                            p_str = "NaN"
                        else:
                            # Ensure p_value is numeric
                            try:
                                p_numeric = float(p_value)
                                p_str = f"<0.0001" if p_numeric < 0.0001 else f"{p_numeric:.4f}"
                            except (ValueError, TypeError):
                                p_numeric = 1.0
                                p_str = str(p_value)
                        
                        correlations[pair_name] = {
                            'r': r_value,
                            'p': p_numeric,
                            'p_str': p_str,
                            'n': n_value
                        }
            
            # Sort by absolute value of correlation coefficient (high to low)
            sorted_pairs = sorted(
                correlations.items(),
                key=lambda x: abs(x[1]['r']) if not pd.isna(x[1]['r']) else 0,
                reverse=True
            )
            
            # Create a row for each correlation pair
            for pair_name, values in sorted_pairs:
                if pd.isna(values['r']):
                    continue
                
                interpretation = self.interpret_correlation(values['r'], values['p'])
                
                report_rows.append({
                    'LLM': llm,
                    'Difficulty': difficulty,
                    'Total_Codes': total_codes,
                    'Correlation_Pair': pair_name,
                    'Pearson_r': round(values['r'], 4),
                    'p_value': values['p_str'],
                    'Sample_Size': int(values['n']),
                    'Interpretation': interpretation,
                    'Abs_r': abs(values['r'])
                })
        
        # Create DataFrame
        report_df = pd.DataFrame(report_rows)
        
        # Sort by LLM, Difficulty, Abs_r
        if not report_df.empty:
            report_df = report_df.sort_values(
                by=['LLM', 'Difficulty', 'Abs_r'],
                ascending=[True, True, False]
            )
            # Remove auxiliary sort column
            report_df = report_df.drop('Abs_r', axis=1)
        
        return report_df
    
    def save_to_excel(self, csv_file: str) -> str:
        """
        Save CSV results as Excel file with two sheets:
        Sheet1: Raw data (wide format)
        Sheet2: Readable report (long format)
        
        Args:
            csv_file: CSV file path
            
        Returns:
            Excel file path, or None if failed
        """
        if not EXCEL_AVAILABLE:
            print("  Skipping Excel generation (openpyxl not installed)")
            return None
        
        try:
            # Read CSV file
            df = pd.read_csv(csv_file)
            
            # Generate readable report
            report_df = self.generate_readable_report(df)
            
            # Generate Excel file path
            excel_file = csv_file.replace('.csv', '.xlsx')
            
            # Create Excel writer
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                # Sheet1: Raw data
                df.to_excel(writer, sheet_name='Raw Data', index=False)
                
                # Sheet2: Readable report
                report_df.to_excel(writer, sheet_name='Readable Report', index=False)
                
                # Get workbook and sheets
                workbook = writer.book
                raw_sheet = writer.sheets['Raw Data']
                report_sheet = writer.sheets['Readable Report']
                
                # Format Sheet1 (raw data)
                # Set header style
                header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
                header_font = Font(bold=True, color="FFFFFF")
                
                for cell in raw_sheet[1]:
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.alignment = Alignment(horizontal='center', vertical='center')
                
                # Auto-adjust column width
                for column in raw_sheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    raw_sheet.column_dimensions[column_letter].width = adjusted_width
                
                # Format Sheet2 (readable report)
                for cell in report_sheet[1]:
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.alignment = Alignment(horizontal='center', vertical='center')
                
                # Adjust column widths
                column_widths = {
                    'A': 15,  # LLM
                    'B': 12,  # Difficulty
                    'C': 12,  # Total_Codes
                    'D': 50,  # Correlation_Pair
                    'E': 12,  # Pearson_r
                    'F': 10,  # p_value
                    'G': 13,  # Sample_Size
                    'H': 45   # Interpretation
                }
                for col, width in column_widths.items():
                    report_sheet.column_dimensions[col].width = width
                
                # Freeze first row
                raw_sheet.freeze_panes = 'A2'
                report_sheet.freeze_panes = 'A2'
            
            print(f"  Excel file generated: {excel_file}")
            return excel_file
            
        except Exception as e:
            print(f"  Error generating Excel file: {e}")
            return None
    
    def save_results_to_csv(self, output_file: str):
        """
        Save analysis results to CSV file
        
        Args:
            output_file: Output file path
        """
        if not self.correlation_results:
            print("No results to save")
            return
        
        # Prepare CSV data
        rows = []
        
        for result in self.correlation_results:
            llm_name = result['llm_name']
            
            # Overall data row
            if result['all']:
                row = {
                    'LLM': llm_name,
                    'Difficulty': 'All',
                }
                row.update(self.format_correlation_for_csv(result['all'].copy()))
                rows.append(row)
            
            # Data rows for each difficulty
            for difficulty in ['easy', 'medium', 'hard']:
                if result[difficulty]:
                    row = {
                        'LLM': llm_name,
                        'Difficulty': difficulty.capitalize(),
                    }
                    row.update(self.format_correlation_for_csv(result[difficulty].copy()))
                    rows.append(row)
        
        # Convert to DataFrame
        df = pd.DataFrame(rows)
        
        # Ensure column order
        base_columns = ['LLM', 'Difficulty', 'N_Codes', 'Total_Codes']
        other_columns = [col for col in df.columns if col not in base_columns]
        ordered_columns = base_columns + sorted(other_columns)
        
        df = df[ordered_columns]
        
        # Save to CSV
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\nAnalysis results saved to: {output_file}")
        
        # Print summary statistics
        print("\n=== Analysis Summary ===")
        print(f"Number of LLMs analyzed: {len(self.correlation_results)}")
        print(f"Total rows: {len(rows)}")
        
        # Print code count summary
        print("\n=== Code Count Summary ===")
        for result in self.correlation_results:
            llm_name = result['llm_name']
            if result['all']:
                n_codes = result['all']['n_codes']
                total_codes = result['all']['total_codes']
                print(f"{llm_name}: {n_codes} codes with issues / {total_codes} total codes")
    
    def run_batch_analysis(self):
        """
        Batch analyze inter-code correlations for all LLMs
        """
        print("=" * 60)
        print("Starting SonarQube Code-Level Correlation Analysis")
        print("=" * 60)
        
        # Check if sonarqube results directory exists
        if not os.path.exists(self.sonarqube_result_dir):
            print(f"Error: {self.sonarqube_result_dir} directory not found")
            return
        
        # Find all *_Code_SonarQube.csv files
        pattern = os.path.join(self.sonarqube_result_dir, "*_Code_SonarQube.csv")
        all_files = glob.glob(pattern)
        
        if not all_files:
            print(f"Error: No *_Code_SonarQube.csv files found in {self.sonarqube_result_dir}")
            return
        
        # Extract ALL unique LLM names (both with and without difficulty suffixes)
        llm_names_set = set()
        difficulty_suffixes = ['_Easy', '_Medium', '_Hard']
        
        for file_path in all_files:
            filename = os.path.basename(file_path)
            # Remove "_Code_SonarQube.csv" suffix
            llm_name = filename.replace("_Code_SonarQube.csv", "")
            
            # Check if it has a difficulty suffix
            base_llm_name = llm_name
            for suffix in difficulty_suffixes:
                if llm_name.endswith(suffix):
                    # Extract base name (without difficulty)
                    base_llm_name = llm_name[:-len(suffix)]
                    break
            
            # Add base LLM name to set
            llm_names_set.add(base_llm_name)
        
        llm_names = sorted(llm_names_set)
        
        print(f"\nFound {len(llm_names)} unique LLMs:")
        for llm in llm_names:
            print(f"  - {llm}")
        
        print("\nStarting analysis...\n")
        
        # Analyze each LLM
        for idx, llm_name in enumerate(llm_names, 1):
            print(f"\n[{idx}/{len(llm_names)}] " + "=" * 50)
            result = self.analyze_llm(llm_name)
            self.correlation_results.append(result)
        
        # Save results
        output_file = os.path.join(self.output_dir, "sonarqube_code_level_summary.csv")
        self.save_results_to_csv(output_file)
        
        # Generate Excel file (if openpyxl is available)
        excel_file = self.save_to_excel(output_file)
        
        print("\n" + "=" * 60)
        print("Code-level correlation analysis complete!")
        print("=" * 60)
        
        return output_file, excel_file


def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("SonarQube Code-Level Correlation Analysis Tool (Fixed)")
    print("=" * 60)
    print("\n✨ Fixed Issues:")
    print("  1. Now correctly calculates correlations between ATTRIBUTES")
    print("     (across all code samples) instead of between FILES")
    print("  2. Uses original column naming convention")
    print("  3. Ensures ALL LLMs are processed (including those with difficulty files)")
    print("\nFeatures:")
    print("  This tool analyzes Pearson correlations between SonarQube results")
    print("  for codes under the same LLM, including code correctness (Passed Tests)")
    print("  Analysis is performed by difficulty classification (easy, medium, hard)")
    print("\n✨ Reads total code counts from sonarqube_summary.csv")
    print("  - Shows both 'codes with issues' and 'total codes'")
    print("  - Identifies missing codes (codes without any issues)")
    print("\nPrerequisites:")
    print("  1. Run sonarqube_checker.py first to generate sonarqube_result/ directory")
    print("  2. sonarqube_result/ directory should contain files:")
    print("     - sonarqube_summary.csv (contains 'LLM' and 'Total Codes' columns)")
    print("     - <LLM>_Code_SonarQube.csv (overall data)")
    print("     - <LLM>_Easy_Code_SonarQube.csv (Easy difficulty)")
    print("     - <LLM>_Medium_Code_SonarQube.csv (Medium difficulty)")
    print("     - <LLM>_Hard_Code_SonarQube.csv (Hard difficulty)")
    print("\nOutput files:")
    print("  - sonarqube_code_level_result/sonarqube_code_level_summary.csv")
    print("  - sonarqube_code_level_result/sonarqube_code_level_summary.xlsx (if openpyxl installed)")
    print("")
    
    analyzer = SonarQubeCorrelationAnalyzer(
        sonarqube_result_dir="sonarqube_result",
        output_dir="sonarqube_code_level_result"
    )
    
    result = analyzer.run_batch_analysis()
    
    if result:
        output_file, excel_file = result
        print(f"\n✓ Analysis complete!")
        print(f"\nGenerated files:")
        print(f"  - {output_file} (CSV format)")
        if excel_file:
            print(f"  - {excel_file} (Excel format with 2 sheets)")
            print(f"    * Sheet1 'Raw Data': Raw data (wide format)")
            print(f"    * Sheet2 'Readable Report': Readable report (long format, sorted by correlation strength)")
        print(f"\nResults description:")
        print(f"  - Each row represents a correlation analysis for an LLM at specific difficulty")
        print(f"  - N_Codes: Number of codes with issues (included in CSV)")
        print(f"  - Total_Codes: Total number of codes (from sonarqube_summary.csv)")
        print(f"  - Missing codes = Total_Codes - N_Codes (codes without any issues)")
        print(f"  - *_r: Pearson correlation coefficient (range: -1 to 1)")
        print(f"     * 1 indicates perfect positive correlation")
        print(f"     * -1 indicates perfect negative correlation")
        print(f"     * 0 indicates no linear correlation")
        print(f"  - *_p-value: Statistical significance")
        print(f"     * < 0.05 indicates statistical significance")
        print(f"     * < 0.01 indicates high significance")
        print(f"  - *_n: Sample size for each correlation pair")
        print(f"  - Correlations are calculated only when code count >= 3")
        print(f"\nCorrelation pairs include:")
        print(f"  - Passed Tests vs all SonarQube metrics")
        print(f"  - Total Issues vs specific issue categories")
        print(f"  - Severity levels (Blocker, Critical, Major, Minor, Info)")
        print(f"  - Issue types (Bug, Vulnerability, Code Smell, Security Hotspot)")
        print(f"  - Clean Code Attributes (Consistency, Intentionality, Adaptability, Responsibility)")
        print(f"  - Software Quality dimensions (Security, Reliability, Maintainability)")
        print(f"  - Cross-category correlations (e.g., Passed Tests vs Bugs, Total Issues vs Maintainability)")
        print(f"\nInterpretation Guide:")
        print(f"  - Strong correlations (|r| >= 0.7): Variables have strong linear relationship")
        print(f"  - Moderate correlations (0.4 <= |r| < 0.7): Variables have moderate linear relationship")
        print(f"  - Weak correlations (|r| < 0.4): Variables have weak or no linear relationship")
        print(f"  - Negative correlations: As one variable increases, the other decreases")
        print(f"  - Positive correlations: Both variables increase or decrease together")
        print(f"\nExample insights:")
        print(f"  - Negative correlation between 'Passed Tests' and 'Total Issues'")
        print(f"    → Codes with more issues tend to fail more tests")
        print(f"  - Positive correlation between 'Type Bug' and 'SQ Reliability'")
        print(f"    → Bugs are strongly associated with reliability issues")
        print(f"  - Correlation between severity levels and software quality dimensions")
        print(f"    → Understanding which issues most impact code quality")
    else:
        print("\nPlease run sonarqube_checker.py first to generate sonarqube_result directory and data files")


if __name__ == "__main__":
    main()
