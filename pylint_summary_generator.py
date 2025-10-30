#!/usr/bin/env python3
import os
import csv
from typing import Dict, List
from collections import defaultdict


class PylintSummaryGenerator:
    def __init__(self):
        """Initialize Pylint summary generator"""
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
        # All data in CSV is valid (original script doesn't write to CSV when encountering empty code)
        if not data:
            # If no data, return empty statistics
            model_name = f"{llm_name} ({difficulty})" if difficulty else llm_name
            return {
                'model': model_name,
                'total_codes': 0,
                'correctness_rate': 0.0,
                'total_issues': 0,
                'avg_issues_per_code': 0.0,
                'convention': 0,
                'refactor': 0,
                'warning': 0,
                'error': 0,
                'fatal': 0,
                'avg_score': 0.0
            }

        # Count statistics
        total_codes = len(data)
        total_issues = 0
        convention_count = 0
        refactor_count = 0
        warning_count = 0
        error_count = 0
        fatal_count = 0

        # For calculating average score
        valid_scores = []
        scored_codes = 0

        # For calculating correctness rate (number of codes with score ≥8)
        correct_codes = 0

        for row in data:
            try:
                # Accumulate issue counts
                total_issues += int(row.get('Total Issues', 0))
                convention_count += int(row.get('Convention (C)', 0))
                refactor_count += int(row.get('Refactor (R)', 0))
                warning_count += int(row.get('Warning (W)', 0))
                error_count += int(row.get('Error (E)', 0))
                fatal_count += int(row.get('Fatal (F)', 0))

                # Process score
                score_str = row.get('Pylint Score', '').strip()
                if score_str and score_str != 'failed' and score_str != 'None' and score_str.lower() != 'none':
                    try:
                        score = float(score_str)
                        valid_scores.append(score)
                        scored_codes += 1

                        # Score ≥8 is considered correct
                        if score >= 8.0:
                            correct_codes += 1
                    except:
                        pass

            except Exception as e:
                print(f"  Error processing data row: {e}")
                continue

        # Calculate averages
        avg_issues = total_issues / total_codes if total_codes > 0 else 0.0
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        # Calculate correctness rate
        correctness_rate = round((correct_codes / total_codes * 100), 2) if total_codes > 0 else 0.0

        # Build model name
        model_name = f"{llm_name} ({difficulty})" if difficulty else llm_name

        return {
            'model': model_name,
            'total_codes': total_codes,
            'correctness_rate': correctness_rate,
            'total_issues': total_issues,
            'avg_issues_per_code': round(avg_issues, 2),
            'convention': convention_count,
            'refactor': refactor_count,
            'warning': warning_count,
            'error': error_count,
            'fatal': fatal_count,
            'avg_score': round(avg_score, 2)
        }

    def analyze_llm_results(self, llm_name: str, result_dir: str = "pylint_result") -> Dict:
        """
        Analyze result CSV files for a single LLM

        Args:
            llm_name: LLM name
            result_dir: pylint_result folder path

        Returns:
            Dictionary containing overall and per-difficulty statistics
        """
        print(f"\nProcessing LLM: {llm_name}")

        results = {}

        # Define CSV file mapping
        csv_files = {
            'all': f"{llm_name}_Code_Pylint.csv",
            'easy': f"{llm_name}_Easy_Code_Pylint.csv",
            'medium': f"{llm_name}_Medium_Code_Pylint.csv",
            'hard': f"{llm_name}_Hard_Code_Pylint.csv"
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
            print(f"    Total issues: {stats['total_issues']}")
            print(f"    Average issues: {stats['avg_issues_per_code']}")
            print(f"    Average score: {stats['avg_score']}/10.0")
            print(f"    Correctness rate: {stats['correctness_rate']}%")

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
                    'Total Issues',
                    'Avg Issues Per Code',
                    'Convention (C)',
                    'Refactor (R)',
                    'Warning (W)',
                    'Error (E)',
                    'Fatal (F)',
                    'Avg Score'
                ])

            # Write data
            for result in self.results:
                writer.writerow([
                    result['model'],
                    result['total_codes'],
                    result['correctness_rate'],
                    result['total_issues'],
                    result['avg_issues_per_code'],
                    result['convention'],
                    result['refactor'],
                    result['warning'],
                    result['error'],
                    result['fatal'],
                    result['avg_score']
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

    def find_llms_in_result_dir(self, result_dir: str = "pylint_result") -> List[str]:
        """
        Find all LLMs in pylint_result folder
        Identify LLMs by finding *_Code_Pylint.csv files (excluding difficulty-specific files)

        Args:
            result_dir: pylint_result folder path

        Returns:
            List of LLM names
        """
        llm_names = set()

        if not os.path.exists(result_dir):
            print(f"Error: {result_dir} folder not found")
            return []

        # Find all *_Code_Pylint.csv files, but exclude those with difficulty markers
        for filename in os.listdir(result_dir):
            if filename.endswith('_Code_Pylint.csv'):
                # Exclude files with difficulty (Easy, Medium, Hard)
                if '_Easy_Code_Pylint.csv' in filename:
                    continue
                if '_Medium_Code_Pylint.csv' in filename:
                    continue
                if '_Hard_Code_Pylint.csv' in filename:
                    continue

                # Extract LLM name
                llm_name = filename.replace('_Code_Pylint.csv', '')
                llm_names.add(llm_name)

        return sorted(list(llm_names))

    def generate_summary(self, result_dir: str = "pylint_result",
                         summary_output: str = "pylint_result/pylint_summary_regenerated.csv"):
        """
        Generate summary file from pylint_result folder

        Args:
            result_dir: pylint_result folder path
            summary_output: Summary results output file
        """
        print("=" * 60)
        print("Starting Summary Generation from CSV Files")
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
        print("Summary Generation Completed!")
        print("=" * 60)
        print(f"\nAll results saved to: {summary_output}")
        print(f"Processed this run: {len(llm_names_to_process)} LLMs")
        print(f"Skipped (completed): {len(skipped_llms)} LLMs")
        print(f"Total: {len(llm_names)} LLMs")


if __name__ == "__main__":
    generator = PylintSummaryGenerator()

    # Generate summary from pylint_result folder
    generator.generate_summary(
        result_dir="pylint_result",  # pylint_result folder path
        summary_output="pylint_result/pylint_summary_regenerated.csv"  # Summary results filename
    )

    print("\nCompleted!")
    print("\nGenerated files:")
    print("  - pylint_result/pylint_summary_regenerated.csv: Summary file regenerated from CSV")