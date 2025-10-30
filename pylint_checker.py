#!/usr/bin/env python3
import json
import os
import csv
import subprocess
import tempfile
import re
from typing import Dict, List, Tuple
import glob
from datetime import datetime


class PylintChecker:
    def __init__(self):
        """Initialize Pylint checker"""
        self.results = []
        self.raw_data_cache = []  # Cache raw data

    def check_pylint_installed(self) -> bool:
        """Check if pylint is installed"""
        try:
            subprocess.run(['pylint', '--version'],
                           capture_output=True,
                           check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def extract_code(self, code_str: str) -> str:
        """
        Extract clean Python code from LLM's response
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
        # Match ```python ... ``` or ``` ... ```
        import re

        # Try to match code blocks starting with ```python or ```
        patterns = [
            r'```python\s*\n(.*?)```',  # ```python ... ```
            r'```\s*\n(.*?)```',  # ``` ... ```
            r'```python\s*(.*?)```',  # ```python...``` (no line break)
            r'```\s*(.*?)```'  # ```...``` (no line break)
        ]

        for pattern in patterns:
            match = re.search(pattern, code_str, re.DOTALL)
            if match:
                return match.group(1).strip()

        # If no code block markers found, return original content (strip whitespace)
        return code_str.strip()

    def save_raw_output(self, llm_name: str, question_id: str, code_index: int,
                        stdout: str, stderr: str, return_code: int):
        """
        Save pylint raw output to cache

        Args:
            llm_name: LLM name
            question_id: Question ID
            code_index: Code index
            stdout: Standard output
            stderr: Standard error output
            return_code: Return code
        """
        raw_entry = {
            'timestamp': datetime.now().isoformat(),
            'question_id': question_id,
            'code_index': code_index,
            'return_code': return_code,
            'stdout': stdout,
            'stderr': stderr
        }
        self.raw_data_cache.append(raw_entry)

    def write_raw_data_to_file(self, llm_name: str, output_dir: str = "pylint_raw"):
        """
        Write cached raw data to file

        Args:
            llm_name: LLM name
            output_dir: Output root directory
        """
        if not self.raw_data_cache:
            return

        # Create output directory structure
        llm_dir = os.path.join(output_dir, llm_name)
        os.makedirs(llm_dir, exist_ok=True)

        # Output file path
        output_file = os.path.join(llm_dir, f"{llm_name}_pylint_raw.txt")

        # Write raw data
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Pylint Raw Output for {llm_name}\n")
            f.write(f"Generated at: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")

            for entry in self.raw_data_cache:
                f.write(f"Timestamp: {entry['timestamp']}\n")
                f.write(f"Question ID: {entry['question_id']}\n")
                f.write(f"Code Index: {entry['code_index']}\n")
                f.write(f"Return Code: {entry['return_code']}\n")
                f.write("-" * 40 + "\n")
                f.write("STDOUT:\n")
                f.write(entry['stdout'] if entry['stdout'] else "(empty)\n")
                f.write("\n" + "-" * 40 + "\n")
                f.write("STDERR:\n")
                f.write(entry['stderr'] if entry['stderr'] else "(empty)\n")
                f.write("\n" + "=" * 80 + "\n\n")

        print(f"  Raw data saved to: {output_file}")

        # Clear cache
        self.raw_data_cache = []

    def run_pylint(self, code: str, question_id: str = None, code_index: int = None,
                   llm_name: str = None) -> Dict:
        """
        Run pylint on a single code snippet

        Args:
            code: Python code
            question_id: Question ID (for recording)
            code_index: Code index (for recording)
            llm_name: LLM name (for recording)

        Returns:
            Issues statistics dictionary
        """
        # First extract clean code
        code = self.extract_code(code)

        if not code or code.strip() == "":
            return {
                'total_issues': 0,
                'convention': 0,
                'refactor': 0,
                'warning': 0,
                'error': 0,
                'fatal': 0,
                'score': None,
                'issues_detail': []
            }

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_file = f.name

        try:
            # First run: Get JSON format issue details
            result_json = subprocess.run(
                ['pylint', '--output-format=json', temp_file],
                capture_output=True,
                text=True
            )

            # Second run: Get text format score information
            result_text = subprocess.run(
                ['pylint', temp_file],
                capture_output=True,
                text=True
            )

            # Save raw output (using JSON version output)
            if llm_name and question_id is not None and code_index is not None:
                self.save_raw_output(
                    llm_name=llm_name,
                    question_id=question_id,
                    code_index=code_index,
                    stdout=result_json.stdout,
                    stderr=result_json.stderr,
                    return_code=result_json.returncode
                )

            # Parse JSON output to get issues list
            score = None
            issues = []

            try:
                issues = json.loads(result_json.stdout)
            except json.JSONDecodeError:
                pass

            # Extract score from text format output
            score_text = result_text.stdout + (result_text.stderr if result_text.stderr else "")

            if score_text:
                # Try to match: "Your code has been rated at 8.57/10"
                match = re.search(r'rated at ([-\d.]+)/10', score_text)
                if match:
                    try:
                        score = float(match.group(1))
                    except:
                        pass

                # If not found, try other formats
                if score is None:
                    match = re.search(r'score: ([-\d.]+)/10', score_text, re.IGNORECASE)
                    if match:
                        try:
                            score = float(match.group(1))
                        except:
                            pass

            # Count issues by type
            type_count = {
                'convention': 0,
                'refactor': 0,
                'warning': 0,
                'error': 0,
                'fatal': 0
            }

            for issue in issues:
                issue_type = issue.get('type', '').lower()
                if issue_type in type_count:
                    type_count[issue_type] += 1

            # If still no score, mark as failed
            if score is None:
                score = "failed"

            return {
                'total_issues': len(issues),
                'convention': type_count['convention'],
                'refactor': type_count['refactor'],
                'warning': type_count['warning'],
                'error': type_count['error'],
                'fatal': type_count['fatal'],
                'score': score,  # Could be float or "failed"
                'issues_detail': issues
            }

        except Exception as e:
            # Save raw output on exception as well
            if llm_name and question_id is not None and code_index is not None:
                self.save_raw_output(
                    llm_name=llm_name,
                    question_id=question_id,
                    code_index=code_index,
                    stdout="",
                    stderr=f"Exception: {str(e)}",
                    return_code=-1
                )

            return {
                'total_issues': 0,
                'convention': 0,
                'refactor': 0,
                'warning': 0,
                'error': 0,
                'fatal': 0,
                'score': None,
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
            List of question info, each element contains question ID and code list
        """
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Check if it's list format (as in your example)
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

    def save_detailed_results(self, llm_name: str, detailed_results: List[Dict],
                              output_dir: str = "pylint_result", difficulty_filter: str = None):
        """
        Save detailed pylint results to separate CSV file

        Args:
            llm_name: LLM name
            detailed_results: Detailed results list
            output_dir: Output directory
            difficulty_filter: Difficulty filter (None, 'easy', 'medium', 'hard')
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Determine filename based on difficulty filter
        if difficulty_filter:
            output_file = os.path.join(output_dir, f"{llm_name}_{difficulty_filter.capitalize()}_Code_Pylint.csv")
            # Filter results for specified difficulty
            filtered_results = [r for r in detailed_results if r['difficulty'].lower() == difficulty_filter.lower()]
        else:
            output_file = os.path.join(output_dir, f"{llm_name}_Code_Pylint.csv")
            filtered_results = detailed_results

        if not filtered_results:
            warning_msg = f"no {difficulty_filter} difficulty codes" if difficulty_filter else "no code"
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
                'Convention (C)',
                'Refactor (R)',
                'Warning (W)',
                'Error (E)',
                'Fatal (F)',
                'Pylint Score',
                'Top Issues',
                'Source Code'
            ])

            # Write detailed results for each code
            for result in filtered_results:
                # Extract top 3 most important issues as summary
                top_issues = []
                for issue in result['issues_detail'][:3]:
                    msg = f"{issue.get('symbol', 'unknown')}:{issue.get('message', '')[:50]}"
                    top_issues.append(msg)
                top_issues_str = " | ".join(top_issues) if top_issues else "None"

                # Handle source code: replace newlines with visible markers for CSV display
                source_code = result.get('source_code', '')
                # Option 1: Keep newlines (will display as multiple lines in Excel)
                # Option 2: Replace newlines with \\n (easy to view but won't actually break lines)
                # Using option 1 here to maintain code readability

                writer.writerow([
                    result['question_id'],
                    result['question_title'],
                    result['difficulty'],
                    result['code_index'],
                    'Yes' if result['passed_tests'] else 'No',
                    result['question_pass@1'],
                    result['total_issues'],
                    result['convention'],
                    result['refactor'],
                    result['warning'],
                    result['error'],
                    result['fatal'],
                    result['score'],
                    top_issues_str,
                    source_code
                ])

        difficulty_str = f"({difficulty_filter})" if difficulty_filter else ""
        print(f"  Detailed results{difficulty_str} saved to: {output_file}")

    def calculate_correctness_rate(self, detailed_results: List[Dict]) -> float:
        """
        Calculate correctness rate: number of codes that passed tests / total number of codes

        Args:
            detailed_results: Detailed results list

        Returns:
            Correctness rate percentage
        """
        if not detailed_results:
            return 0.0

        # Calculate number of codes that passed tests
        passed_count = sum(1 for result in detailed_results if result['passed_tests'])
        total_codes = len(detailed_results)

        return round((passed_count / total_codes * 100), 2) if total_codes > 0 else 0.0

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

        # Clear raw data cache
        self.raw_data_cache = []

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
                        'convention': 0,
                        'refactor': 0,
                        'warning': 0,
                        'error': 0,
                        'fatal': 0,
                        'total_score': 0.0,
                        'valid_codes': 0,
                        'scored_codes': 0
                    }

                for idx, code in enumerate(code_list):
                    if code and code.strip():
                        print(f"    Checking {question_id} code [{idx + 1}/{len(code_list)}]...", end='\r')

                        # Run pylint check (pass extra parameters for recording raw data)
                        result = self.run_pylint(
                            code=code,
                            question_id=question_id,
                            code_index=idx + 1,
                            llm_name=llm_name
                        )

                        # Update difficulty statistics
                        stats = difficulty_stats[difficulty]
                        stats['total_issues'] += result['total_issues']
                        stats['convention'] += result['convention']
                        stats['refactor'] += result['refactor']
                        stats['warning'] += result['warning']
                        stats['error'] += result['error']
                        stats['fatal'] += result['fatal']
                        stats['valid_codes'] += 1

                        # Only count valid scores (not "failed")
                        if result['score'] != "failed" and result['score'] is not None:
                            stats['total_score'] += result['score']
                            stats['scored_codes'] += 1

                        # Save detailed result
                        detailed_result = {
                            'question_id': question_id,
                            'question_title': question_title,
                            'difficulty': difficulty,
                            'code_index': idx + 1,
                            'passed_tests': graded_list[idx] if idx < len(graded_list) else False,
                            'question_pass@1': pass_at_1,
                            'total_issues': result['total_issues'],
                            'convention': result['convention'],
                            'refactor': result['refactor'],
                            'warning': result['warning'],
                            'error': result['error'],
                            'fatal': result['fatal'],
                            'score': result['score'],
                            'issues_detail': result['issues_detail'],
                            'source_code': code  # Save raw code
                        }
                        all_detailed_results.append(detailed_result)

        print(f"\n  Check completed: {len(all_detailed_results)} valid codes")

        # Save raw data to file
        self.write_raw_data_to_file(llm_name)

        # Save detailed results for all codes
        self.save_detailed_results(llm_name, all_detailed_results, output_dir="pylint_result")

        # Save detailed results by difficulty
        for difficulty in difficulty_stats.keys():
            self.save_detailed_results(llm_name, all_detailed_results,
                                       output_dir="pylint_result",
                                       difficulty_filter=difficulty)

        # Calculate overall statistics and statistics by difficulty
        results = {}

        # Overall statistics
        total_codes = sum(stats['valid_codes'] for stats in difficulty_stats.values())
        total_issues = sum(stats['total_issues'] for stats in difficulty_stats.values())
        total_convention = sum(stats['convention'] for stats in difficulty_stats.values())
        total_refactor = sum(stats['refactor'] for stats in difficulty_stats.values())
        total_warning = sum(stats['warning'] for stats in difficulty_stats.values())
        total_error = sum(stats['error'] for stats in difficulty_stats.values())
        total_fatal = sum(stats['fatal'] for stats in difficulty_stats.values())
        total_score_sum = sum(stats['total_score'] for stats in difficulty_stats.values())
        total_scored_codes = sum(stats['scored_codes'] for stats in difficulty_stats.values())

        avg_issues = total_issues / total_codes if total_codes > 0 else 0
        avg_score = total_score_sum / total_scored_codes if total_scored_codes > 0 else 0
        correctness_rate = self.calculate_correctness_rate(all_detailed_results)

        results['all'] = {
            'model': llm_name,
            'total_codes': total_codes,
            'total_issues': total_issues,
            'avg_issues_per_code': round(avg_issues, 2),
            'convention': total_convention,
            'refactor': total_refactor,
            'warning': total_warning,
            'error': total_error,
            'fatal': total_fatal,
            'avg_score': round(avg_score, 2),
            'scored_codes': total_scored_codes,
            'correctness_rate': correctness_rate
        }

        print(f"\n  Overall statistics:")
        print(f"    Code count: {total_codes}")
        print(f"    Total issues: {total_issues}")
        print(f"    Average issues: {round(avg_issues, 2)}")
        print(f"    Average score: {round(avg_score, 2)}/10.0")
        print(f"    Correctness rate: {correctness_rate}%")

        # Statistics by difficulty
        for difficulty, stats in difficulty_stats.items():
            valid_codes = stats['valid_codes']
            if valid_codes == 0:
                continue

            avg_issues = stats['total_issues'] / valid_codes
            avg_score = stats['total_score'] / stats['scored_codes'] if stats['scored_codes'] > 0 else 0

            # Calculate correctness rate for this difficulty
            difficulty_results = [r for r in all_detailed_results if r['difficulty'] == difficulty]
            correctness_rate = self.calculate_correctness_rate(difficulty_results)

            results[difficulty] = {
                'model': f"{llm_name} ({difficulty})",
                'total_codes': valid_codes,
                'total_issues': stats['total_issues'],
                'avg_issues_per_code': round(avg_issues, 2),
                'convention': stats['convention'],
                'refactor': stats['refactor'],
                'warning': stats['warning'],
                'error': stats['error'],
                'fatal': stats['fatal'],
                'avg_score': round(avg_score, 2),
                'scored_codes': stats['scored_codes'],
                'correctness_rate': correctness_rate
            }

            print(f"\n  {difficulty.capitalize()} difficulty statistics:")
            print(f"    Code count: {valid_codes}")
            print(f"    Total issues: {stats['total_issues']}")
            print(f"    Average issues: {round(avg_issues, 2)}")
            print(f"    Average score: {round(avg_score, 2)}/10.0")
            print(f"    Correctness rate: {correctness_rate}%")

        return results

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

        # Check if file exists and if header needs to be written
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

    def run_batch_analysis(self, submission_dir="submissions", summary_output="pylint_result/pylint_summary.csv"):
        """
        Batch analyze all LLM results in the submissions folder

        Args:
            submission_dir: Submissions folder path
            summary_output: Summary results output file
        """
        # Check if pylint is installed
        if not self.check_pylint_installed():
            print("Error: Pylint is not installed")
            print("Please run: pip install pylint")
            return

        print("Pylint is installed, starting code quality check...")

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

        # Filter out LLMs to be processed (exclude completed ones)
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
        print("Starting batch code quality check")
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
                print("\n  Current LLM statistical results:")
                for result in self.results:
                    print(f"    {result['model']}: codes={result['total_codes']}, "
                          f"correctness rate={result['correctness_rate']}%, "
                          f"average score={result['avg_score']}/10.0")

        # Final summary
        print("\n" + "=" * 60)
        print("Code quality check completed!")
        print("=" * 60)
        print(f"\nAll results saved to: {summary_output}")
        print(f"Processed this run: {len(llm_folders_to_process)} LLMs")
        print(f"Skipped (completed): {len(skipped_llms)} LLMs")
        print(f"Total: {len(llm_folders)} LLMs")


if __name__ == "__main__":
    checker = PylintChecker()

    # Run batch analysis
    checker.run_batch_analysis(
        submission_dir="submissions",  # Submissions folder path
        summary_output="pylint_result/pylint_summary.csv"  # Summary results filename
    )

    print("\nCompleted!")
    print("\nGenerated files:")
    print("  - pylint_result/pylint_summary.csv: Summary results for all LLMs (including overall and by difficulty)")
    print("  - pylint_result/<LLM>_Code_Pylint.csv: Detailed evaluation for all codes of each LLM")
    print("  - pylint_result/<LLM>_Easy_Code_Pylint.csv: Detailed evaluation for Easy difficulty codes of each LLM")
    print("  - pylint_result/<LLM>_Medium_Code_Pylint.csv: Detailed evaluation for Medium difficulty codes of each LLM")
    print("  - pylint_result/<LLM>_Hard_Code_Pylint.csv: Detailed evaluation for Hard difficulty codes of each LLM")
    print("  - pylint_raw/<LLM>/<LLM>_pylint_raw.txt: Raw pylint output data for each LLM")