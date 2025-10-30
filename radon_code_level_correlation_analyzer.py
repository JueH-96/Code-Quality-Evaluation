#!/usr/bin/env python3
"""
Radon Code-Level Correlation Analyzer (Enhanced Version)
Analyzes Pearson correlations between radon results for codes under the same LLM
Analyzes by difficulty classification and includes test pass rate correlations
- Skips empty codes (no Source Code)
- Normalizes columns by non-empty code count (averages)
- Calculates accuracy statistics
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


class RadonCorrelationAnalyzer:
    def __init__(self, radon_result_dir="radon_result", output_dir="radon_code_level_result"):
        """
        Initialize correlation analyzer
        
        Args:
            radon_result_dir: radon results directory
            output_dir: output directory
        """
        self.radon_result_dir = radon_result_dir
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Store analysis results
        self.correlation_results = []
    
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
            print(f"  Read {original_count} codes, all used for analysis")
            
            # Convert Passed Tests to numeric format immediately after reading
            if 'Passed Tests' in df.columns:
                df = self._convert_passed_tests_to_numeric(df)
            
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
            
            # Try direct numeric conversion
            try:
                return float(value_str)
            except:
                return np.nan
        
        df['Passed Tests'] = df['Passed Tests'].apply(parse_passed_tests)
        print(f"    Passed Tests converted to numeric. Sample values: {df['Passed Tests'].head(5).tolist()}")
        non_null_values = df['Passed Tests'].dropna()
        if len(non_null_values) > 0:
            print(f"    Passed Tests range: [{non_null_values.min():.2f}, {non_null_values.max():.2f}]")
            print(f"    Passed Tests: {(non_null_values == 1.0).sum()} passed, {(non_null_values == 0.0).sum()} failed")
        
        return df
    
    def _normalize_columns_by_total_codes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize columns by dividing by total number of NON-EMPTY rows (codes) to get averages
        Empty codes are determined by checking if 'Source Code' column is empty/NaN
        Only applies to columns that don't contain 'avg' (case-insensitive)
        
        Args:
            df: DataFrame with columns to normalize
            
        Returns:
            DataFrame with normalized columns
        """
        # Get total number of codes (rows, excluding header)
        total_rows = len(df)
        
        if total_rows == 0:
            print(f"    Warning: DataFrame is empty. Skipping normalization.")
            return df
        
        # Make a copy to avoid modifying original
        df_normalized = df.copy()
        
        # Columns to potentially normalize - Radon metrics
        columns_to_check = [
            'Complexity',
            'Functions',
            'LOC',
            'LLOC',
            'SLOC',
            'Comments',
            'Blank Lines'
        ]
        
        # Find columns that exist in the dataframe
        existing_cols = [col for col in columns_to_check if col in df.columns]
        
        if not existing_cols:
            print(f"    Warning: No columns found to normalize. Skipping normalization.")
            return df
        
        # Identify non-empty codes based on Source Code column
        if 'Source Code' in df.columns:
            # A code is non-empty if Source Code column has actual content
            # Handle NaN, empty strings, whitespace, and newlines
            def is_non_empty_code(value):
                if pd.isna(value):
                    return False
                value_str = str(value).strip()
                # Empty string or just whitespace/newlines
                if value_str == '' or value_str == '\n' or value_str.replace('\n', '').strip() == '':
                    return False
                return True
            
            non_empty_mask = df['Source Code'].apply(is_non_empty_code)
            non_empty_count = non_empty_mask.sum()
            empty_count = total_rows - non_empty_count
            
            print(f"    Total rows: {total_rows}")
            print(f"    Non-empty codes (with Source Code): {non_empty_count}")
            print(f"    Empty codes (no Source Code, skipped): {empty_count}")
        else:
            # Fallback: if Source Code column doesn't exist, use numeric columns
            print(f"    Warning: 'Source Code' column not found. Using numeric columns to identify empty codes.")
            non_empty_mask = (df[existing_cols].fillna(0) != 0).any(axis=1)
            non_empty_count = non_empty_mask.sum()
            empty_count = total_rows - non_empty_count
            
            print(f"    Total rows: {total_rows}")
            print(f"    Non-empty codes: {non_empty_count}")
            print(f"    Empty codes (skipped): {empty_count}")
        
        if non_empty_count == 0:
            print(f"    Warning: All codes are empty. Skipping normalization.")
            return df
        
        # Normalize by non-empty code count
        normalized_count = 0
        for col in columns_to_check:
            # Check if column exists and doesn't contain 'avg' (case-insensitive)
            if col in df.columns and 'avg' not in col.lower():
                try:
                    # Divide by non-empty code count to get average per non-empty code
                    df_normalized[col] = df[col] / non_empty_count
                    normalized_count += 1
                    print(f"    Normalized column: {col} (divided by {non_empty_count} non-empty codes)")
                except Exception as e:
                    print(f"    Warning: Failed to normalize {col}: {e}")
        
        if normalized_count > 0:
            print(f"    Total {normalized_count} columns normalized by non-empty code count")
        else:
            print(f"    No columns were normalized (either already contain 'avg' or not found)")
        
        return df_normalized
    
    def calculate_accuracy_stats(self, df: pd.DataFrame) -> Dict:
        """
        Calculate accuracy statistics from Passed Tests column
        
        Args:
            df: DataFrame containing Passed Tests column
            
        Returns:
            Dictionary with accuracy statistics
        """
        stats_dict = {}
        
        if 'Passed Tests' not in df.columns:
            return stats_dict
        
        # Get non-null values
        passed_tests = df['Passed Tests'].dropna()
        
        if len(passed_tests) == 0:
            return stats_dict
        
        # Calculate statistics
        total_codes = len(passed_tests)
        passed_count = (passed_tests == 1.0).sum()
        failed_count = (passed_tests == 0.0).sum()
        
        # Calculate accuracy (pass rate)
        accuracy = passed_count / total_codes if total_codes > 0 else 0.0
        
        stats_dict['Total_Codes'] = total_codes
        stats_dict['Passed_Count'] = passed_count
        stats_dict['Failed_Count'] = failed_count
        stats_dict['Accuracy'] = round(accuracy, 4)
        stats_dict['Accuracy_Percent'] = f"{accuracy * 100:.2f}%"
        
        return stats_dict
    
    def calculate_correlations(self, df: pd.DataFrame) -> Dict:
        """
        Calculate Pearson correlation coefficients between columns in dataframe
        Includes Passed Tests analysis, columns normalized by non-empty code count
        
        Args:
            df: DataFrame containing code evaluation data
            
        Returns:
            Dictionary of correlation coefficients and accuracy statistics
        """
        if df.empty or len(df) < 3:
            # Too little data to calculate meaningful correlations
            return {}
        
        # Print all available columns for debugging
        print(f"    Available columns in CSV: {list(df.columns)}")
        
        # First, determine non-empty codes count for normalization
        total_rows = len(df)
        
        # Identify non-empty codes based on Source Code column
        if 'Source Code' in df.columns:
            # A code is non-empty if Source Code column has actual content
            def is_non_empty_code(value):
                if pd.isna(value):
                    return False
                value_str = str(value).strip()
                if value_str == '' or value_str == '\n' or value_str.replace('\n', '').strip() == '':
                    return False
                return True
            
            non_empty_mask = df['Source Code'].apply(is_non_empty_code)
            non_empty_count = non_empty_mask.sum()
            
            # Filter to only non-empty codes for accuracy calculation
            df_non_empty = df[non_empty_mask].copy()
            print(f"    Total rows: {total_rows}, Non-empty codes: {non_empty_count}, Empty codes: {total_rows - non_empty_count}")
        else:
            # If no Source Code column, use all rows
            print(f"    Warning: 'Source Code' column not found. Using all rows.")
            non_empty_count = total_rows
            df_non_empty = df.copy()
        
        # Calculate accuracy statistics using only non-empty codes
        accuracy_stats = self.calculate_accuracy_stats(df_non_empty)
        # Override Total_Codes with non-empty count
        accuracy_stats['Total_Codes'] = non_empty_count
        print(f"    Accuracy: {accuracy_stats.get('Accuracy_Percent', 'N/A')} ({accuracy_stats.get('Passed_Count', 0)}/{non_empty_count})")
        
        # Normalize columns by non-empty code count before correlation calculation
        df = self._normalize_columns_by_total_codes(df)
        
        # Select numeric columns for analysis - Radon metrics
        numeric_columns = [
            'Passed Tests',
            'Complexity',
            'Avg Complexity',
            'Functions',
            'Maintainability Index',
            'LOC',
            'LLOC',
            'SLOC',
            'Comments',
            'Blank Lines'
        ]
        
        # Ensure these columns exist
        available_columns = [col for col in numeric_columns if col in df.columns]
        
        print(f"    Columns used for correlation (after normalization): {available_columns}")
        
        if not available_columns:
            return accuracy_stats
        
        # Extract data
        data = df[available_columns]
        
        # Calculate correlation coefficient matrix
        corr_results = {}
        
        # Calculate Pearson correlation coefficient for each pair of variables
        for i, col1 in enumerate(available_columns):
            for col2 in available_columns[i+1:]:
                try:
                    # Use all data (remove NaN values)
                    valid_data = data[[col1, col2]].dropna()
                    
                    if len(valid_data) < 3:
                        continue
                    
                    # Skip if both columns have no variance
                    if valid_data[col1].std() == 0 or valid_data[col2].std() == 0:
                        continue
                    
                    # Calculate Pearson correlation coefficient and p-value
                    r, p_value = stats.pearsonr(valid_data[col1], valid_data[col2])
                    
                    # Use full names as keys
                    key = f"{col1}-{col2}"
                    corr_results[key] = {
                        'r': r,
                        'p_value': p_value,
                        'n': len(valid_data)
                    }
                except Exception as e:
                    print(f"    Error calculating {col1} vs {col2} correlation: {e}")
                    continue
        
        print(f"    Calculated {len(corr_results)} correlation pairs")
        
        # Merge accuracy stats with correlation results
        corr_results.update(accuracy_stats)
        
        return corr_results
    
    def analyze_llm(self, llm_name: str) -> Dict:
        """
        Analyze inter-code correlations for a single LLM
        
        Args:
            llm_name: LLM name
            
        Returns:
            Analysis results dictionary
        """
        print(f"\nAnalyzing {llm_name}...")
        
        results = {
            'llm_name': llm_name,
            'all': {},
            'easy': {},
            'medium': {},
            'hard': {}
        }
        
        # Analyze overall data
        all_file = os.path.join(self.radon_result_dir, f"{llm_name}_Code_Radon.csv")
        if os.path.exists(all_file):
            df_all = self.read_llm_code_data(all_file)
            if not df_all.empty and len(df_all) >= 3:
                print(f"  Overall: {len(df_all)} valid codes")
                results['all'] = self.calculate_correlations(df_all)
                results['all']['n_codes'] = len(df_all)
            else:
                print(f"  Overall: Insufficient data, skipping")
        
        # Analyze data for each difficulty
        for difficulty in ['easy', 'medium', 'hard']:
            difficulty_file = os.path.join(
                self.radon_result_dir, 
                f"{llm_name}_{difficulty.capitalize()}_Code_Radon.csv"
            )
            
            if os.path.exists(difficulty_file):
                df_diff = self.read_llm_code_data(difficulty_file)
                if not df_diff.empty and len(df_diff) >= 3:
                    print(f"  {difficulty.capitalize()}: {len(df_diff)} valid codes")
                    results[difficulty] = self.calculate_correlations(df_diff)
                    results[difficulty]['n_codes'] = len(df_diff)
                else:
                    print(f"  {difficulty.capitalize()}: Insufficient data, skipping")
        
        return results
    
    def format_correlation_for_csv(self, corr_dict: Dict) -> Dict:
        """
        Format correlation dictionary as CSV row
        
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
        formatted['N_Codes'] = n_codes
        
        # Extract accuracy statistics
        formatted['Total_Codes'] = corr_dict.pop('Total_Codes', n_codes)
        formatted['Passed_Count'] = corr_dict.pop('Passed_Count', 0)
        formatted['Failed_Count'] = corr_dict.pop('Failed_Count', 0)
        formatted['Accuracy'] = corr_dict.pop('Accuracy', 0.0)
        formatted['Accuracy_Percent'] = corr_dict.pop('Accuracy_Percent', '0.00%')
        
        # Iterate through each correlation pair
        for pair_name, values in corr_dict.items():
            if isinstance(values, dict) and 'r' in values:
                # Use full column names directly
                base_name = pair_name.replace('-', ' vs ')
                
                formatted[f"{base_name} (r)"] = round(values['r'], 4)
                formatted[f"{base_name} (p-value)"] = round(values['p_value'], 4) if values['p_value'] >= 0.0001 else '<0.0001'
                # Add sample size information
                formatted[f"{base_name} (n)"] = values.get('n', n_codes)
        
        return formatted
    
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
            total_codes = row.get('Total_Codes', n_codes)  # Use Total_Codes if available
            accuracy = row.get('Accuracy_Percent', 'N/A')
            
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
                    'Total_Codes': total_codes,  # Use total_codes (non-empty code count)
                    'Accuracy': accuracy,
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
    
    def save_to_excel(self, csv_file: str):
        """
        Save CSV results as Excel file with two sheets:
        Sheet1: Raw data (wide format)
        Sheet2: Readable report (long format)
        
        Args:
            csv_file: CSV file path
        """
        if not EXCEL_AVAILABLE:
            print("  Skipping Excel file generation (requires openpyxl installation)")
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
                # Write Sheet1: Raw data
                df.to_excel(writer, sheet_name='Raw Data', index=False)
                
                # Write Sheet2: Readable report
                report_df.to_excel(writer, sheet_name='Readable Report', index=False)
                
                # Get workbook and sheets
                workbook = writer.book
                raw_sheet = workbook['Raw Data']
                report_sheet = workbook['Readable Report']
                
                # Define styles
                header_font = Font(bold=True, size=11, color="FFFFFF")
                header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
                
                # Format Sheet1 (raw data)
                for cell in raw_sheet[1]:
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.alignment = Alignment(horizontal='center', vertical='center')
                
                # Adjust column widths for Sheet1
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
                
                # Adjust column widths for Sheet2
                column_widths = {
                    'A': 15,  # LLM
                    'B': 12,  # Difficulty
                    'C': 12,  # Total_Codes
                    'D': 12,  # Accuracy
                    'E': 50,  # Correlation_Pair (wider for full names)
                    'F': 12,  # Pearson_r
                    'G': 10,  # p_value
                    'H': 13,  # Sample_Size
                    'I': 45   # Interpretation
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
            import traceback
            print(f"  Detailed error: {traceback.format_exc()}")
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
                row.update(self.format_correlation_for_csv(result['all']))
                rows.append(row)
            
            # Data rows for each difficulty
            for difficulty in ['easy', 'medium', 'hard']:
                if result[difficulty]:
                    row = {
                        'LLM': llm_name,
                        'Difficulty': difficulty.capitalize(),
                    }
                    row.update(self.format_correlation_for_csv(result[difficulty]))
                    rows.append(row)
        
        # Convert to DataFrame
        df = pd.DataFrame(rows)
        
        # Ensure column order - put accuracy stats right after difficulty
        base_columns = ['LLM', 'Difficulty', 'N_Codes', 'Total_Codes', 'Passed_Count', 'Failed_Count', 'Accuracy', 'Accuracy_Percent']
        other_columns = [col for col in df.columns if col not in base_columns]
        ordered_columns = base_columns + sorted(other_columns)
        
        # Only keep columns that exist in df
        ordered_columns = [col for col in ordered_columns if col in df.columns]
        
        df = df[ordered_columns]
        
        # Save to CSV
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\nAnalysis results saved to: {output_file}")
        
        # Print summary statistics
        print("\n=== Analysis Summary ===")
        print(f"Number of LLMs analyzed: {len(self.correlation_results)}")
        print(f"Total rows: {len(rows)}")
    
    def run_batch_analysis(self):
        """
        Batch analyze inter-code correlations for all LLMs
        """
        print("=" * 60)
        print("Starting Radon Code-Level Correlation Analysis")
        print("(With normalization by row count and accuracy statistics)")
        print("=" * 60)
        
        # Check if radon results directory exists
        if not os.path.exists(self.radon_result_dir):
            print(f"Error: {self.radon_result_dir} directory not found")
            return
        
        # Find all *_Code_Radon.csv files (overall files)
        pattern = os.path.join(self.radon_result_dir, "*_Code_Radon.csv")
        all_files = glob.glob(pattern)
        
        if not all_files:
            print(f"Error: No *_Code_Radon.csv files found in {self.radon_result_dir}")
            return
        
        # Extract LLM names, excluding files with difficulty suffixes
        llm_names = []
        difficulty_suffixes = ['_Easy', '_Medium', '_Hard']
        
        for file_path in all_files:
            filename = os.path.basename(file_path)
            # Remove "_Code_Radon.csv" suffix
            llm_name = filename.replace("_Code_Radon.csv", "")
            
            # Check if contains difficulty suffix
            has_difficulty_suffix = any(llm_name.endswith(suffix) for suffix in difficulty_suffixes)
            
            # Only keep LLMs without difficulty suffix
            if not has_difficulty_suffix:
                llm_names.append(llm_name)
        
        llm_names = sorted(set(llm_names))
        
        print(f"\nFound {len(llm_names)} LLMs:")
        for llm in llm_names:
            print(f"  - {llm}")
        
        print("\nStarting analysis...\n")
        
        # Analyze each LLM
        for idx, llm_name in enumerate(llm_names, 1):
            print(f"\n[{idx}/{len(llm_names)}] " + "=" * 50)
            result = self.analyze_llm(llm_name)
            self.correlation_results.append(result)
        
        # Save results
        output_file = os.path.join(self.output_dir, "radon_code_level_summary.csv")
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
    print("Radon Code-Level Correlation Analysis Tool")
    print("(Enhanced Version with Normalization and Accuracy Stats)")
    print("=" * 60)
    print("\nFeatures:")
    print("  This tool analyzes Pearson correlations between radon results")
    print("  for codes under the same LLM, including code correctness (Passed Tests)")
    print("  Analysis is performed by difficulty classification (easy, medium, hard)")
    print("  ** Skips empty codes (no Source Code) **")
    print("  ** Uses full column names instead of abbreviations **")
    print("  ** Calculates accuracy statistics (pass rate) **")
    print("  ** Columns not containing 'avg' are normalized by non-empty code count **")
    print("\nPrerequisites:")
    print("  1. Run radon_checker.py first to generate radon_result/ directory")
    print("  2. radon_result/ directory should contain files:")
    print("     - <LLM>_Code_Radon.csv (overall data)")
    print("     - <LLM>_Easy_Code_Radon.csv (Easy difficulty)")
    print("     - <LLM>_Medium_Code_Radon.csv (Medium difficulty)")
    print("     - <LLM>_Hard_Code_Radon.csv (Hard difficulty)")
    print("\nOutput files:")
    print("  - radon_code_level_result/radon_code_level_summary.csv")
    print("  - radon_code_level_result/radon_code_level_summary.xlsx (if openpyxl installed)")
    print("")
    
    analyzer = RadonCorrelationAnalyzer(
        radon_result_dir="radon_result",
        output_dir="radon_code_level_result"
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
        print(f"  - N_Codes: Total number of codes (including empty)")
        print(f"  - Total_Codes: Number of non-empty codes used for analysis")
        print(f"  - Accuracy: Code correctness rate (pass rate) for non-empty codes")
        print(f"  - ** Full column names used throughout (e.g., 'Passed Tests vs Complexity') **")
        print(f"  - ** Values are normalized (averaged) by non-empty code count before correlation **")
        print(f"  - *_r: Pearson correlation coefficient (range: -1 to 1)")
        print(f"     * 1 indicates perfect positive correlation")
        print(f"     * -1 indicates perfect negative correlation")
        print(f"     * 0 indicates no linear correlation")
        print(f"  - *_p: p-value")
        print(f"     * < 0.05 indicates statistical significance")
        print(f"     * < 0.01 indicates high significance")
        print(f"  - Correlations are calculated only when code count >= 3")
        print(f"\nCorrelation pairs include:")
        print(f"  - Passed Tests vs all code complexity metrics (correctness vs quality)")
        print(f"  - Complexity vs Maintainability Index")
        print(f"  - LOC/LLOC/SLOC inter-correlations (code size metrics)")
        print(f"  - Comments and Blank Lines vs other metrics (documentation patterns)")
        print(f"  - Avg Complexity vs Functions (function-level complexity)")
        print(f"\nKey Research Questions:")
        print(f"  - Does code correctness correlate with complexity?")
        print(f"  - Is maintainability related to code size?")
        print(f"  - Do more complex codes have better/worse test pass rates?")
        print(f"  - Are comments correlated with code quality metrics?")
    else:
        print("\nPlease run radon_checker.py first to generate radon_result directory and data files")


if __name__ == "__main__":
    main()
