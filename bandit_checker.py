#!/usr/bin/env python3
import json
import os
import csv
import tempfile
import re
from typing import Dict, List, Tuple
import glob
import subprocess
from datetime import datetime


class BanditChecker:
    def __init__(self):
        """Initialize Bandit checker"""
        self.results = []
        self.raw_data_buffer = []  # Buffer for caching raw data

    def check_bandit_installed(self) -> bool:
        """Check if bandit is installed"""
        try:
            subprocess.run(['bandit', '--version'],
                           capture_output=True,
                           check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def extract_code(self, code_str: str) -> str:
        """
        Extract clean Python code from LLM responses
        Remove markdown code block markers and other non-code content

        Args:
            code_str: Code string that may contain markdown markers

        Returns:
            Clean Python code
        """
        if not code_str:
            return ""

        # Remove \r and other special characters
        code_str = code_str.replace('\\r', '').replace('\\n', '\n')

        # Find markdown code blocks
        patterns = [
            r'```python\s*\n(.*?)```',  # ```python ... ```
            r'```\s*\n(.*?)```',  # ``` ... ```
            r'```python\s*(.*?)```',  # ```python...``` (no newline)
            r'```\s*(.*?)```'  # ```...``` (no newline)
        ]

        for pattern in patterns:
            match = re.search(pattern, code_str, re.DOTALL)
            if match:
                return match.group(1).strip()

        # If no code block markers found, return original content (trimmed)
        return code_str.strip()

    def save_raw_data(self, llm_name: str, output_dir: str = "bandit_raw"):
        """
        Save raw bandit output data to file

        Args:
            llm_name: LLM name
            output_dir: Output directory
        """
        # Ensure output directory exists
        llm_output_dir = os.path.join(output_dir, llm_name)
        os.makedirs(llm_output_dir, exist_ok=True)

        # Generate filename (using LLM name)
        output_file = os.path.join(llm_output_dir, f"{llm_name}_raw.txt")

        # Write raw data
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Bandit Raw Data for {llm_name}\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            for entry in self.raw_data_buffer:
                f.write(f"Question ID: {entry['question_id']}\n")
                f.write(f"Question Title: {entry['question_title']}\n")
                f.write(f"Difficulty: {entry['difficulty']}\n")
                f.write(f"Code Index: {entry['code_index']}\n")
                f.write(f"Timestamp: {entry['timestamp']}\n")
                f.write("-" * 80 + "\n")

                # Write raw bandit output
                f.write("BANDIT SECURITY SCAN OUTPUT:\n")
                f.write(entry['bandit_output'])
                f.write("\n")

                f.write("=" * 80 + "\n\n")

        print(f"  Raw data saved to: {output_file}")

    def run_bandit(self, code: str, question_id: str = "", question_title: str = "",
                   difficulty: str = "", code_index: int = 0) -> Dict:
        """
        Run bandit security check on a single code snippet

        Args:
            code: Python code
            question_id: Question ID (for logging)
            question_title: Question title (for logging)
            difficulty: Difficulty level (for logging)
            code_index: Code index (for logging)

        Returns:
            Security issue statistics dictionary
        """
        # First extract clean code
        code = self.extract_code(code)

        if not code or code.strip() == "":
            return {
                'total_issues': 0,
                'high': 0,
                'medium': 0,
                'low': 0,
                'undefined': 0,
                'confidence_high': 0,
                'confidence_medium': 0,
                'confidence_low': 0,
                'issues_detail': []
            }

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_file = f.name

        try:
            # Run bandit with JSON output format
            result = subprocess.run(
                ['bandit', '-f', 'json', temp_file],
                capture_output=True,
                text=True,
                timeout=10
            )

            # Save raw output (if question info provided)
            if question_id:
                raw_entry = {
                    'question_id': question_id,
                    'question_title': question_title,
                    'difficulty': difficulty,
                    'code_index': code_index,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'bandit_output': result.stdout if result.stdout else "No output"
                }
                self.raw_data_buffer.append(raw_entry)

            # Parse JSON output
            try:
                data = json.loads(result.stdout)
                results = data.get('results', [])

                # Count issues by severity
                severity_count = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'UNDEFINED': 0}
                confidence_count = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'UNDEFINED': 0}

                for issue in results:
                    severity = issue.get('issue_severity', 'UNDEFINED')
                    confidence = issue.get('issue_confidence', 'UNDEFINED')

                    if severity in severity_count:
                        severity_count[severity] += 1
                    else:
                        severity_count['UNDEFINED'] += 1

                    if confidence in confidence_count:
                        confidence_count[confidence] += 1
                    else:
                        confidence_count['UNDEFINED'] += 1

                return {
                    'total_issues': len(results),
                    'high': severity_count['HIGH'],
                    'medium': severity_count['MEDIUM'],
                    'low': severity_count['LOW'],
                    'undefined': severity_count['UNDEFINED'],
                    'confidence_high': confidence_count['HIGH'],
                    'confidence_medium': confidence_count['MEDIUM'],
                    'confidence_low': confidence_count['LOW'],
                    'issues_detail': results
                }

            except (json.JSONDecodeError, KeyError):
                return {
                    'total_issues': 0,
                    'high': 0,
                    'medium': 0,
                    'low': 0,
                    'undefined': 0,
                    'confidence_high': 0,
                    'confidence_medium': 0,
                    'confidence_low': 0,
                    'issues_detail': []
                }

        except subprocess.TimeoutExpired:
            return {
                'total_issues': 0,
                'high': 0,
                'medium': 0,
                'low': 0,
                'undefined': 0,
                'confidence_high': 0,
                'confidence_medium': 0,
                'confidence_low': 0,
                'issues_detail': []
            }
        except Exception as e:
            return {
                'total_issues': 0,
                'high': 0,
                'medium': 0,
                'low': 0,
                'undefined': 0,
                'confidence_high': 0,
                'confidence_medium': 0,
                'confidence_low': 0,
                'issues_detail': []
            }
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def read_json_file(self, json_file: str) -> List[Dict]:
        """
        Read question and code information from JSON file

        Args:
            json_file: JSON file path

        Returns:
            List of question information, each element contains question ID and code list
        """
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Check if it's in list format
            if isinstance(data, list):
                questions = []
                for item in data:
                    question_info = {
                        'question_id': item.get('question_id', 'unknown'),
                        'question_title': item.get('question_title', 'unknown'),
                        'difficulty': item.get('difficulty', 'unknown'),
                        'code_list': item.get('code_list', []),
                        'graded_list': item.get('graded_list', []),
                        'pass@1': item.get('pass@1', 0.0)
                    }
                    questions.append(question_info)
                return questions
            else:
                return []

        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            return []

    def calculate_correctness_rate(self, detailed_results: List[Dict]) -> float:
        """
        Calculate correctness rate: number of codes that passed tests / total number of codes

        Args:
            detailed_results: List of detailed results

        Returns:
            Correctness rate percentage
        """
        if not detailed_results:
            return 0.0

        # Count codes that passed tests
        passed_count = sum(1 for result in detailed_results if result['passed_tests'])
        total_codes = len(detailed_results)

        return round((passed_count / total_codes * 100), 2) if total_codes > 0 else 0.0

    def save_detailed_results(self, llm_name: str, detailed_results: List[Dict],
                              output_dir: str = "bandit_result", difficulty_filter: str = None):
        """
        Save detailed bandit results for each code to a separate CSV file

        Args:
            llm_name: LLM name
            detailed_results: List of detailed results
            output_dir: Output directory
            difficulty_filter: Difficulty filter (None, 'easy', 'medium', 'hard')
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Determine filename based on difficulty filter
        if difficulty_filter:
            output_file = os.path.join(output_dir, f"{llm_name}_{difficulty_filter.capitalize()}_Code_Bandit.csv")
            # Filter results for specified difficulty
            filtered_results = [r for r in detailed_results if r['difficulty'].lower() == difficulty_filter.lower()]
        else:
            output_file = os.path.join(output_dir, f"{llm_name}_Code_Bandit.csv")
            filtered_results = detailed_results

        if not filtered_results:
            warning_msg = f"no {difficulty_filter} difficulty codes" if difficulty_filter else "no codes"
            print(f"  Warning: {warning_msg}")
            return

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow([
                'Question ID',
                'Question Title',
                'Difficulty',
                'Code Index',
                'Passed Tests',
                'Question Pass@1',
                'Total Issues',
                'High Severity',
                'Medium Severity',
                'Low Severity',
                'High Confidence',
                'Medium Confidence',
                'Low Confidence',
                'Top Issues',
                'Source Code'
            ])

            # Write detailed results for each code
            for result in filtered_results:
                # Extract top 3 most important issues as summary
                top_issues = []
                for issue in result['issues_detail'][:3]:
                    test_id = issue.get('test_id', 'unknown')
                    issue_text = issue.get('issue_text', '')[:50]
                    msg = f"{test_id}:{issue_text}"
                    top_issues.append(msg)
                top_issues_str = " | ".join(top_issues) if top_issues else "None"

                # Get source code
                source_code = result.get('source_code', '')

                writer.writerow([
                    result['question_id'],
                    result['question_title'],
                    result['difficulty'],
                    result['code_index'],
                    'Yes' if result['passed_tests'] else 'No',
                    result['question_pass@1'],
                    result['total_issues'],
                    result['high'],
                    result['medium'],
                    result['low'],
                    result['confidence_high'],
                    result['confidence_medium'],
                    result['confidence_low'],
                    top_issues_str,
                    source_code
                ])

        difficulty_str = f"({difficulty_filter})" if difficulty_filter else ""
        print(f"  Detailed results{difficulty_str} saved to: {output_file}")

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
                    base_name = llm_name.split('(')[0].strip() if '(' in llm_name else llm_name
                    if base_name:
                        completed_llms.add(base_name)

            if completed_llms:
                print(f"\nFound already processed LLMs: {', '.join(sorted(completed_llms))}")

        except Exception as e:
            print(f"Error reading existing summary file: {e}")

        return completed_llms

    def analyze_llm_folder(self, llm_folder: str) -> Dict:
        """
        Analyze all JSON files in a single LLM folder

        Args:
            llm_folder: LLM folder path

        Returns:
            Analysis result dictionary
        """
        llm_name = os.path.basename(llm_folder)
        print(f"\nAnalyzing LLM: {llm_name}")
        print(f"Folder: {llm_folder}")

        # Clear raw data buffer
        self.raw_data_buffer = []

        # Find JSON files in folder
        json_files = glob.glob(os.path.join(llm_folder, "*.json"))

        if not json_files:
            print(f"  Warning: No JSON files found")
            return None

        print(f"  Found {len(json_files)} JSON files")

        # Store all detailed results
        all_detailed_results = []

        # Statistics grouped by difficulty
        difficulty_stats = {}

        # Statistics
        total_code_count = 0  # Total number of codes (including empty codes)
        skipped_code_count = 0  # Number of empty codes skipped
        processed_code_count = 0  # Number of codes actually processed

        # Process each JSON file
        for json_file in json_files:
            print(f"  Processing file: {os.path.basename(json_file)}")
            questions = self.read_json_file(json_file)

            for question in questions:
                question_id = question['question_id']
                question_title = question['question_title']
                difficulty = question.get('difficulty', 'unknown').lower()
                code_list = question['code_list']
                graded_list = question['graded_list']
                pass_at_1 = question.get('pass@1', 0.0)

                # Initialize difficulty statistics
                if difficulty not in difficulty_stats:
                    difficulty_stats[difficulty] = {
                        'total_issues': 0,
                        'high': 0,
                        'medium': 0,
                        'low': 0,
                        'undefined': 0,
                        'confidence_high': 0,
                        'confidence_medium': 0,
                        'confidence_low': 0,
                        'valid_codes': 0,
                        'codes_with_issues': 0
                    }

                # [FIX] Iterate through all codes without skipping any
                for idx, code in enumerate(code_list):
                    total_code_count += 1

                    # Check if code is empty or only whitespace
                    is_empty = not code or not code.strip()

                    if is_empty:
                        skipped_code_count += 1
                        print(f"    Skipped {question_id} code [{idx + 1}/{len(code_list)}] (empty code)")

                        # [FIX] Even for empty code, record in detailed results to maintain index consistency
                        detailed_result = {
                            'question_id': question_id,
                            'question_title': question_title,
                            'difficulty': difficulty,
                            'code_index': idx + 1,
                            'passed_tests': graded_list[idx] if idx < len(graded_list) else False,
                            'question_pass@1': pass_at_1,
                            'total_issues': 0,
                            'high': 0,
                            'medium': 0,
                            'low': 0,
                            'confidence_high': 0,
                            'confidence_medium': 0,
                            'confidence_low': 0,
                            'issues_detail': [],
                            'source_code': code if code else ""  # Save original code (even if empty)
                        }
                        all_detailed_results.append(detailed_result)
                        continue

                    # Process non-empty code
                    processed_code_count += 1
                    print(f"    Checking {question_id} code [{idx + 1}/{len(code_list)}]...", end='\r')

                    # Run bandit check (pass question info for logging)
                    result = self.run_bandit(code, question_id, question_title, difficulty, idx + 1)

                    # Update difficulty statistics
                    stats = difficulty_stats[difficulty]
                    stats['total_issues'] += result['total_issues']
                    stats['high'] += result['high']
                    stats['medium'] += result['medium']
                    stats['low'] += result['low']
                    stats['undefined'] += result['undefined']
                    stats['confidence_high'] += result['confidence_high']
                    stats['confidence_medium'] += result['confidence_medium']
                    stats['confidence_low'] += result['confidence_low']
                    stats['valid_codes'] += 1

                    if result['total_issues'] > 0:
                        stats['codes_with_issues'] += 1

                    # Save detailed result
                    detailed_result = {
                        'question_id': question_id,
                        'question_title': question_title,
                        'difficulty': difficulty,
                        'code_index': idx + 1,
                        'passed_tests': graded_list[idx] if idx < len(graded_list) else False,
                        'question_pass@1': pass_at_1,
                        'total_issues': result['total_issues'],
                        'high': result['high'],
                        'medium': result['medium'],
                        'low': result['low'],
                        'confidence_high': result['confidence_high'],
                        'confidence_medium': result['confidence_medium'],
                        'confidence_low': result['confidence_low'],
                        'issues_detail': result['issues_detail'],
                        'source_code': code  # Save original code
                    }
                    all_detailed_results.append(detailed_result)

        # Output statistics
        print(f"\n  Code statistics:")
        print(f"    Total codes: {total_code_count}")
        print(f"    Actually processed: {processed_code_count}")
        print(f"    Skipped (empty code): {skipped_code_count}")
        print(f"    Detailed result entries: {len(all_detailed_results)}")

        # Save raw bandit output data
        self.save_raw_data(llm_name, output_dir="bandit_raw")

        # Save detailed results for all codes
        self.save_detailed_results(llm_name, all_detailed_results, output_dir="bandit_result")

        # Save detailed results by difficulty
        for difficulty in difficulty_stats.keys():
            self.save_detailed_results(llm_name, all_detailed_results,
                                       output_dir="bandit_result",
                                       difficulty_filter=difficulty)

        # Calculate overall statistics and statistics by difficulty
        results = {}

        # Overall statistics
        total_codes = sum(stats['valid_codes'] for stats in difficulty_stats.values())
        total_issues = sum(stats['total_issues'] for stats in difficulty_stats.values())
        total_high = sum(stats['high'] for stats in difficulty_stats.values())
        total_medium = sum(stats['medium'] for stats in difficulty_stats.values())
        total_low = sum(stats['low'] for stats in difficulty_stats.values())
        total_undefined = sum(stats['undefined'] for stats in difficulty_stats.values())
        total_conf_high = sum(stats['confidence_high'] for stats in difficulty_stats.values())
        total_conf_medium = sum(stats['confidence_medium'] for stats in difficulty_stats.values())
        total_conf_low = sum(stats['confidence_low'] for stats in difficulty_stats.values())
        total_codes_with_issues = sum(stats['codes_with_issues'] for stats in difficulty_stats.values())

        issue_rate = (total_codes_with_issues / total_codes * 100) if total_codes > 0 else 0
        avg_issues = total_issues / total_codes if total_codes > 0 else 0
        correctness_rate = self.calculate_correctness_rate(all_detailed_results)

        results['all'] = {
            'model': llm_name,
            'total_codes': total_codes,
            'correctness_rate': correctness_rate,
            'codes_with_issues': total_codes_with_issues,
            'issue_rate': round(issue_rate, 2),
            'total_issues': total_issues,
            'avg_issues_per_code': round(avg_issues, 2),
            'high_severity': total_high,
            'medium_severity': total_medium,
            'low_severity': total_low,
            'undefined_severity': total_undefined,
            'high_confidence': total_conf_high,
            'medium_confidence': total_conf_medium,
            'low_confidence': total_conf_low
        }

        print(f"\n  Overall statistics:")
        print(f"    Valid codes: {total_codes}")
        print(f"    Correctness rate: {correctness_rate}%")
        print(f"    Codes with security issues: {total_codes_with_issues} ({round(issue_rate, 2)}%)")
        print(f"    Total security issues: {total_issues}")
        print(f"    High: {total_high}, Medium: {total_medium}, Low: {total_low}")

        # Statistics by difficulty
        for difficulty, stats in difficulty_stats.items():
            valid_codes = stats['valid_codes']
            if valid_codes == 0:
                continue

            issue_rate = (stats['codes_with_issues'] / valid_codes * 100) if valid_codes > 0 else 0
            avg_issues = stats['total_issues'] / valid_codes if valid_codes > 0 else 0

            # Calculate correctness rate for this difficulty
            difficulty_results = [r for r in all_detailed_results if r['difficulty'] == difficulty]
            correctness_rate = self.calculate_correctness_rate(difficulty_results)

            results[difficulty] = {
                'model': f"{llm_name} ({difficulty})",
                'total_codes': valid_codes,
                'correctness_rate': correctness_rate,
                'codes_with_issues': stats['codes_with_issues'],
                'issue_rate': round(issue_rate, 2),
                'total_issues': stats['total_issues'],
                'avg_issues_per_code': round(avg_issues, 2),
                'high_severity': stats['high'],
                'medium_severity': stats['medium'],
                'low_severity': stats['low'],
                'undefined_severity': stats['undefined'],
                'high_confidence': stats['confidence_high'],
                'medium_confidence': stats['confidence_medium'],
                'low_confidence': stats['confidence_low']
            }

            print(f"\n  {difficulty.capitalize()} difficulty statistics:")
            print(f"    Codes: {valid_codes}")
            print(f"    Correctness rate: {correctness_rate}%")
            print(f"    Codes with security issues: {stats['codes_with_issues']} ({round(issue_rate, 2)}%)")
            print(f"    Total security issues: {stats['total_issues']}")
            print(f"    High: {stats['high']}, Medium: {stats['medium']}, Low: {stats['low']}")

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

        # Check if file exists and whether to write header
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

    def run_batch_analysis(self, submission_dir="submissions", summary_output="bandit_result/bandit_summary.csv"):
        """
        Batch analyze all LLM results in the submissions folder

        Args:
            submission_dir: Submissions folder path
            summary_output: Summary results output file
        """
        # Check if bandit is installed
        if not self.check_bandit_installed():
            print("Error: Bandit is not installed")
            print("Please run: pip install bandit")
            return

        print("Bandit is installed, starting security check...")

        # Check if submissions folder exists
        if not os.path.exists(submission_dir):
            print(f"Error: {submission_dir} folder not found")
            return

        # Read list of completed LLMs
        completed_llms = self.read_existing_summary(summary_output)

        # Get all LLM subfolders
        llm_folders = [
            os.path.join(submission_dir, d)
            for d in os.listdir(submission_dir)
            if os.path.isdir(os.path.join(submission_dir, d))
        ]

        if not llm_folders:
            print(f"Error: No subfolders found in {submission_dir}")
            return

        # Filter out LLMs that need processing (exclude completed ones)
        llm_folders_to_process = []
        skipped_llms = []

        for folder in llm_folders:
            llm_name = os.path.basename(folder)
            if llm_name in completed_llms:
                skipped_llms.append(llm_name)
            else:
                llm_folders_to_process.append(folder)

        print(f"\nFound {len(llm_folders)} LLM folders:")
        for folder in llm_folders:
            llm_name = os.path.basename(folder)
            status = "✓ Completed" if llm_name in completed_llms else "○ Pending"
            print(f"  {status} {llm_name}")

        if skipped_llms:
            print(f"\nSkipping already processed LLMs ({len(skipped_llms)}): {', '.join(skipped_llms)}")

        if not llm_folders_to_process:
            print("\nAll LLMs have been processed, nothing to do!")
            return

        print(f"\nNeed to process {len(llm_folders_to_process)} LLMs")
        print("\n" + "=" * 60)
        print("Starting batch security check")
        print("=" * 60)

        # Determine if append mode is needed (if there are completed LLMs, use append mode)
        append_mode = len(completed_llms) > 0

        # Analyze each LLM folder that needs processing
        for idx, llm_folder in enumerate(llm_folders_to_process, 1):
            llm_name = os.path.basename(llm_folder)
            print(f"\n[{idx}/{len(llm_folders_to_process)}] " + "=" * 50)
            results = self.analyze_llm_folder(llm_folder)

            if results:
                # Clear temporary results list
                self.results = []

                # Add overall results
                self.results.append(results['all'])
                # Add results by difficulty
                for difficulty in ['easy', 'medium', 'hard']:
                    if difficulty in results:
                        self.results.append(results[difficulty])

                # Save to CSV immediately
                self.save_summary_to_csv(summary_output, append=append_mode)

                # After first write, use append mode for subsequent writes
                if not append_mode:
                    append_mode = True

                # Print statistics for current LLM
                print("\n  Current LLM statistics:")
                for result in self.results:
                    print(f"    {result['model']}: codes={result['total_codes']}, "
                          f"correctness={result['correctness_rate']}%, "
                          f"issue_rate={result['issue_rate']}%, "
                          f"high={result['high_severity']}")

        # Final summary
        print("\n" + "=" * 60)
        print("Security check completed!")
        print("=" * 60)
        print(f"\nAll results saved to: {summary_output}")
        print(f"Processed this run: {len(llm_folders_to_process)} LLMs")
        print(f"Skipped (completed): {len(skipped_llms)} LLMs")
        print(f"Total: {len(llm_folders)} LLMs")


if __name__ == "__main__":
    checker = BanditChecker()

    # Run batch analysis
    checker.run_batch_analysis(
        submission_dir="submissions",  # Submissions folder path
        summary_output="bandit_result/bandit_summary.csv"  # Summary results filename
    )

    print("\nCompleted!")
    print("\nGenerated files:")
    print("  - bandit_result/bandit_summary.csv: Summary results for all LLMs (including overall and by difficulty)")
    print("  - bandit_result/<LLM>_Code_Bandit.csv: Detailed evaluation for all codes of each LLM")
    print("  - bandit_result/<LLM>_Easy_Code_Bandit.csv: Detailed evaluation for Easy difficulty codes of each LLM")
    print("  - bandit_result/<LLM>_Medium_Code_Bandit.csv: Detailed evaluation for Medium difficulty codes of each LLM")
    print("  - bandit_result/<LLM>_Hard_Code_Bandit.csv: Detailed evaluation for Hard difficulty codes of each LLM")
    print("  - bandit_raw/<LLM>/<LLM>_raw.txt: Raw bandit output data for each LLM")