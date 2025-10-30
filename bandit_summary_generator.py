#!/usr/bin/env python3
import os
import csv
from typing import Dict, List


class BanditSummaryGenerator:
    def __init__(self):
        """Initialize Bandit summary generator"""
        self.results = []

    def read_csv_file(self, csv_file: str) -> List[Dict]:
        """
        Read CSV file and return data list

        Args:
            csv_file: CSV file path

        Returns:
            List of data dictionaries
        """
        data = []

        if not os.path.exists(csv_file):
            return data

        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append(row)
        except Exception as e:
            print(f"  Error reading CSV file {csv_file}: {e}")

        return data

    def is_empty_code_result(self, row: Dict) -> bool:
        """
        Determine if this is an empty code result

        Empty code characteristics (corresponding to extract_code returning empty in original script):
        - Total Issues is 0
        - All severity levels are 0
        - All confidence levels are 0
        - Source Code is empty or only whitespace

        Args:
            row: A row of data from CSV file

        Returns:
            True if empty code result, False otherwise
        """
        try:
            # Check if source code is empty
            source_code = row.get('Source Code', '').strip()
            if not source_code:
                return True

            # Check if all metrics are 0 (empty code characteristic)
            total_issues = int(row.get('Total Issues', 0))
            high = int(row.get('High Severity', 0))
            medium = int(row.get('Medium Severity', 0))
            low = int(row.get('Low Severity', 0))
            high_conf = int(row.get('High Confidence', 0))
            medium_conf = int(row.get('Medium Confidence', 0))
            low_conf = int(row.get('Low Confidence', 0))

            # If all metrics are 0 and source code is empty, consider it empty code
            if (total_issues == 0 and high == 0 and medium == 0 and low == 0 and
                    high_conf == 0 and medium_conf == 0 and low_conf == 0 and not source_code):
                return True

            return False

        except Exception as e:
            # If parsing error occurs, handle conservatively, don't skip
            return False

    def calculate_statistics(self, data: List[Dict], llm_name: str, difficulty: str = None) -> Dict:
        """
        Calculate statistics

        Args:
            data: CSV data list
            llm_name: LLM name
            difficulty: Difficulty level (None means all difficulties)

        Returns:
            Statistics result dictionary
        """
        # Filter out empty code results (consistent with pylint/radon)
        valid_data = [row for row in data if not self.is_empty_code_result(row)]

        if not valid_data:
            # If no valid data, return empty statistics
            model_name = f"{llm_name} ({difficulty})" if difficulty else llm_name
            return {
                'model': model_name,
                'total_codes': 0,
                'correctness_rate': 0.0,
                'codes_with_issues': 0,
                'issue_rate': 0.0,
                'total_issues': 0,
                'avg_issues_per_code': 0.0,
                'high_severity': 0,
                'medium_severity': 0,
                'low_severity': 0,
                'undefined_severity': 0,
                'high_confidence': 0,
                'medium_confidence': 0,
                'low_confidence': 0
            }

        # Count statistics (only for non-empty code)
        total_codes = len(valid_data)
        total_issues = 0
        high_severity = 0
        medium_severity = 0
        low_severity = 0
        undefined_severity = 0
        high_confidence = 0
        medium_confidence = 0
        low_confidence = 0

        # For calculating correctness rate (number of codes that passed tests)
        correct_codes = 0

        # For calculating issue rate (number of codes with security issues)
        codes_with_issues = 0

        for row in valid_data:
            try:
                # Process passed tests flag
                passed = row.get('Passed Tests', 'No').strip().lower()
                if passed == 'yes':
                    correct_codes += 1

                # Accumulate issue count
                issues = int(row.get('Total Issues', 0))
                total_issues += issues

                # If has issues, count in codes_with_issues
                if issues > 0:
                    codes_with_issues += 1

                # Accumulate severity levels
                high_severity += int(row.get('High Severity', 0))
                medium_severity += int(row.get('Medium Severity', 0))
                low_severity += int(row.get('Low Severity', 0))

                # Accumulate confidence levels
                high_confidence += int(row.get('High Confidence', 0))
                medium_confidence += int(row.get('Medium Confidence', 0))
                low_confidence += int(row.get('Low Confidence', 0))

            except Exception as e:
                print(f"  Error processing data row: {e}")
                continue

        # Calculate averages
        avg_issues = total_issues / total_codes if total_codes > 0 else 0.0

        # Calculate correctness rate
        correctness_rate = round((correct_codes / total_codes * 100), 2) if total_codes > 0 else 0.0

        # Calculate issue rate
        issue_rate = round((codes_with_issues / total_codes * 100), 2) if total_codes > 0 else 0.0

        # Build model name
        model_name = f"{llm_name} ({difficulty})" if difficulty else llm_name

        return {
            'model': model_name,
            'total_codes': total_codes,
            'correctness_rate': correctness_rate,
            'codes_with_issues': codes_with_issues,
            'issue_rate': issue_rate,
            'total_issues': total_issues,
            'avg_issues_per_code': round(avg_issues, 2),
            'high_severity': high_severity,
            'medium_severity': medium_severity,
            'low_severity': low_severity,
            'undefined_severity': undefined_severity,
            'high_confidence': high_confidence,
            'medium_confidence': medium_confidence,
            'low_confidence': low_confidence
        }

    def analyze_llm_results(self, llm_name: str, result_dir: str = "bandit_result") -> Dict:
        """
        Analyze result CSV files for a single LLM

        Args:
            llm_name: LLM name
            result_dir: bandit_result folder path

        Returns:
            Dictionary containing overall and per-difficulty statistics
        """
        print(f"\nProcessing LLM: {llm_name}")

        results = {}

        # Define CSV file mapping
        csv_files = {
            'all': f"{llm_name}_Code_Bandit.csv",
            'easy': f"{llm_name}_Easy_Code_Bandit.csv",
            'medium': f"{llm_name}_Medium_Code_Bandit.csv",
            'hard': f"{llm_name}_Hard_Code_Bandit.csv"
        }

        # Read and analyze each CSV file
        for key, filename in csv_files.items():
            csv_path = os.path.join(result_dir, filename)

            if not os.path.exists(csv_path):
                print(f"  Warning: File not found {filename}")
                continue

            # Read CSV data
            data = self.read_csv_file(csv_path)

            if not data:
                print(f"  Warning: {filename} is empty or cannot be read")
                continue

            # Calculate statistics
            difficulty = None if key == 'all' else key
            stats = self.calculate_statistics(data, llm_name, difficulty)

            results[key] = stats

            # Print statistics
            if key == 'all':
                print(f"  Overall statistics:")
            else:
                print(f"  {key.capitalize()} difficulty statistics:")

            print(f"    Codes: {stats['total_codes']}")
            print(f"    Correctness rate: {stats['correctness_rate']}%")
            print(f"    Codes with security issues: {stats['codes_with_issues']} ({stats['issue_rate']}%)")
            print(f"    Total security issues: {stats['total_issues']}")
            print(
                f"    High: {stats['high_severity']}, Medium: {stats['medium_severity']}, Low: {stats['low_severity']}")

        return results

    def save_summary_to_csv(self, output_file: str, append: bool = False):
        """
        Save summary results to CSV file

        Args:
            output_file: Output CSV file path
            append: Whether to use append mode (True for append, False for overwrite)
        """
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Check if file exists and whether header needs to be written
        file_exists = os.path.exists(output_file)
        write_header = not file_exists or not append

        mode = 'a' if append and file_exists else 'w'

        with open(output_file, mode, newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Write header (only when needed)
            if write_header:
                writer.writerow([
                    'LLM',
                    'Total Codes',
                    'Correctness Rate (%)',
                    'Codes With Issues',
                    'Issue Rate (%)',
                    'Total Issues',
                    'Avg Issues Per Code',
                    'High Severity',
                    'Medium Severity',
                    'Low Severity',
                    'Undefined Severity',
                    'High Confidence',
                    'Medium Confidence',
                    'Low Confidence'
                ])

            # Write data
            for result in self.results:
                writer.writerow([
                    result['model'],
                    result['total_codes'],
                    result['correctness_rate'],
                    result['codes_with_issues'],
                    result['issue_rate'],
                    result['total_issues'],
                    result['avg_issues_per_code'],
                    result['high_severity'],
                    result['medium_severity'],
                    result['low_severity'],
                    result['undefined_severity'],
                    result['high_confidence'],
                    result['medium_confidence'],
                    result['low_confidence']
                ])

        if not append:
            print(f"\nSummary results saved to: {output_file}")
        else:
            print(f"\nSummary results appended to: {output_file}")

    def read_existing_summary(self, summary_file: str) -> set:
        """
        Read existing summary file and get list of already processed LLMs

        Args:
            summary_file: Summary CSV file path

        Returns:
            Set of already processed LLM names
        """
        completed_llms = set()

        if not os.path.exists(summary_file):
            return completed_llms

        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    llm_name = row.get('LLM', '').strip()
                    # Extract base LLM name (remove difficulty markers)
                    # Example: "GPT4 (easy)" -> "GPT4"
                    base_name = llm_name.split('(')[0].strip() if '(' in llm_name else llm_name
                    if base_name:
                        completed_llms.add(base_name)

            if completed_llms:
                print(f"\nFound already processed LLMs: {', '.join(sorted(completed_llms))}")

        except Exception as e:
            print(f"Error reading existing summary file: {e}")

        return completed_llms

    def find_llms_in_result_dir(self, result_dir: str = "bandit_result") -> List[str]:
        """
        Find all LLMs in bandit_result folder
        Identify LLMs by finding *_Code_Bandit.csv files (excluding difficulty-specific files)

        Args:
            result_dir: bandit_result folder path

        Returns:
            List of LLM names
        """
        llm_names = set()

        if not os.path.exists(result_dir):
            print(f"Error: {result_dir} folder not found")
            return []

        # Find all *_Code_Bandit.csv files, but exclude those with difficulty markers
        for filename in os.listdir(result_dir):
            if filename.endswith('_Code_Bandit.csv'):
                # Exclude files with difficulty (Easy, Medium, Hard)
                if '_Easy_Code_Bandit.csv' in filename:
                    continue
                if '_Medium_Code_Bandit.csv' in filename:
                    continue
                if '_Hard_Code_Bandit.csv' in filename:
                    continue

                # Extract LLM name
                llm_name = filename.replace('_Code_Bandit.csv', '')
                llm_names.add(llm_name)

        return sorted(list(llm_names))

    def generate_summary(self, result_dir: str = "bandit_result",
                         summary_output: str = "bandit_result/bandit_summary_regenerated.csv"):
        """
        Generate summary file from bandit_result folder

        Args:
            result_dir: bandit_result folder path
            summary_output: Summary results output file
        """
        print("=" * 60)
        print("Starting Bandit Summary Generation from CSV Files")
        print("=" * 60)

        # Find all LLMs
        llm_names = self.find_llms_in_result_dir(result_dir)

        if not llm_names:
            print(f"Error: No LLM CSV files found in {result_dir} folder")
            return

        print(f"\nFound {len(llm_names)} LLMs:")
        for llm_name in llm_names:
            print(f"  - {llm_name}")

        # Read list of completed LLMs (if avoiding duplicates)
        completed_llms = self.read_existing_summary(summary_output)

        # Filter out LLMs that need processing
        llm_names_to_process = []
        skipped_llms = []

        for llm_name in llm_names:
            if llm_name in completed_llms:
                skipped_llms.append(llm_name)
            else:
                llm_names_to_process.append(llm_name)

        if skipped_llms:
            print(f"\nSkipping already processed LLMs ({len(skipped_llms)}): {', '.join(skipped_llms)}")

        if not llm_names_to_process:
            print("\nAll LLMs have been processed, nothing to do!")
            return

        print(f"\nNeed to process {len(llm_names_to_process)} LLMs")
        print("\n" + "=" * 60)

        # Determine if append mode is needed
        append_mode = len(completed_llms) > 0

        # Process each LLM
        for idx, llm_name in enumerate(llm_names_to_process, 1):
            print(f"\n[{idx}/{len(llm_names_to_process)}] " + "=" * 50)
            results = self.analyze_llm_results(llm_name, result_dir)

            if results:
                # Clear temporary results list
                self.results = []

                # Add overall results
                if 'all' in results:
                    self.results.append(results['all'])

                # Add per-difficulty results
                for difficulty in ['easy', 'medium', 'hard']:
                    if difficulty in results:
                        self.results.append(results[difficulty])

                # Save to CSV immediately
                self.save_summary_to_csv(summary_output, append=append_mode)

                # After first write, use append mode for subsequent writes
                if not append_mode:
                    append_mode = True

        # Final summary
        print("\n" + "=" * 60)
        print("Bandit Summary Generation Completed!")
        print("=" * 60)
        print(f"\nAll results saved to: {summary_output}")
        print(f"Processed this run: {len(llm_names_to_process)} LLMs")
        print(f"Skipped (completed): {len(skipped_llms)} LLMs")
        print(f"Total: {len(llm_names)} LLMs")


if __name__ == "__main__":
    generator = BanditSummaryGenerator()

    # Generate summary from bandit_result folder
    generator.generate_summary(
        result_dir="bandit_result",  # bandit_result folder path
        summary_output="bandit_result/bandit_summary_regenerated.csv"  # Summary results filename
    )

    print("\nCompleted!")
    print("\nGenerated files:")
    print("  - bandit_result/bandit_summary_regenerated.csv: Summary file regenerated from CSV")