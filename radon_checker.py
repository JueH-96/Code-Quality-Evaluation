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


class RadonChecker:
    def __init__(self):
        """Initialize Radon checker"""
        self.results = []
        self.raw_data_buffer = []  # Buffer for raw data

    def check_radon_installed(self) -> bool:
        """Check if radon is installed"""
        try:
            subprocess.run(['radon', '--version'],
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

    def save_raw_data(self, llm_name: str, output_dir: str = "radon_raw"):
        """
        Save raw radon output data to file

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
            f.write(f"Radon Raw Data for {llm_name}\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            for entry in self.raw_data_buffer:
                f.write(f"Question ID: {entry['question_id']}\n")
                f.write(f"Question Title: {entry['question_title']}\n")
                f.write(f"Difficulty: {entry['difficulty']}\n")
                f.write(f"Code Index: {entry['code_index']}\n")
                f.write(f"Timestamp: {entry['timestamp']}\n")
                f.write("-" * 80 + "\n")

                # Write cyclomatic complexity raw output
                f.write("CYCLOMATIC COMPLEXITY (radon cc):\n")
                f.write(entry['cc_output'])
                f.write("\n")

                # Write maintainability index raw output
                f.write("MAINTAINABILITY INDEX (radon mi):\n")
                f.write(entry['mi_output'])
                f.write("\n")

                # Write raw metrics output
                f.write("RAW METRICS (radon raw):\n")
                f.write(entry['raw_output'])
                f.write("\n")

                f.write("=" * 80 + "\n\n")

        print(f"  Raw data saved to: {output_file}")

    def run_radon_cc(self, code: str, question_id: str = "", question_title: str = "",
                     difficulty: str = "", code_index: int = 0) -> Dict:
        """
        Run Radon cyclomatic complexity analysis

        Args:
            code: Python code
            question_id: Question ID (for recording)
            question_title: Question title (for recording)
            difficulty: Difficulty (for recording)
            code_index: Code index (for recording)

        Returns:
            Complexity statistics dictionary
        """
        # First extract clean code
        code = self.extract_code(code)

        if not code or code.strip() == "":
            return {'complexity': 0, 'avg_complexity': 0, 'grade': 'A', 'functions': 0}

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_file = f.name

        try:
            # Run radon cc (cyclomatic complexity)
            result = subprocess.run(
                ['radon', 'cc', '-s', '-j', temp_file],
                capture_output=True,
                text=True,
                timeout=10
            )

            # Save raw output (if question info provided)
            if question_id:
                if not hasattr(self, '_current_raw_entry'):
                    self._current_raw_entry = {}
                self._current_raw_entry['cc_output'] = result.stdout if result.stdout else "No output"

            # Parse JSON output
            try:
                data = json.loads(result.stdout)
                if temp_file in data:
                    functions = data[temp_file]
                    if functions:
                        total_complexity = sum(f.get('complexity', 0) for f in functions)
                        avg_complexity = total_complexity / len(functions) if functions else 0
                        # Get worst complexity grade
                        grades = [f.get('rank', 'A') for f in functions]
                        worst_grade = self.get_worst_grade(grades)

                        return {
                            'complexity': total_complexity,
                            'avg_complexity': round(avg_complexity, 2),
                            'grade': worst_grade,
                            'functions': len(functions)
                        }
            except (json.JSONDecodeError, KeyError):
                pass

            return {'complexity': 0, 'avg_complexity': 0, 'grade': 'A', 'functions': 0}

        except subprocess.TimeoutExpired:
            return {'complexity': 0, 'avg_complexity': 0, 'grade': 'A', 'functions': 0}
        except Exception as e:
            return {'complexity': 0, 'avg_complexity': 0, 'grade': 'A', 'functions': 0}
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def run_radon_mi(self, code: str, question_id: str = "") -> float:
        """
        Run Radon maintainability index analysis

        Args:
            code: Python code
            question_id: Question ID (for recording)

        Returns:
            Maintainability index (0-100)
        """
        # First extract clean code
        code = self.extract_code(code)

        if not code or code.strip() == "":
            return 0.0

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_file = f.name

        try:
            # Run radon mi (maintainability index)
            result = subprocess.run(
                ['radon', 'mi', '-s', '-j', temp_file],
                capture_output=True,
                text=True,
                timeout=10
            )

            # Save raw output (if question info provided)
            if question_id and hasattr(self, '_current_raw_entry'):
                self._current_raw_entry['mi_output'] = result.stdout if result.stdout else "No output"

            # Parse JSON output
            try:
                data = json.loads(result.stdout)
                if temp_file in data:
                    mi_data = data[temp_file]
                    return mi_data.get('mi', 0.0)
            except (json.JSONDecodeError, KeyError):
                pass

            return 0.0

        except subprocess.TimeoutExpired:
            return 0.0
        except Exception as e:
            return 0.0
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def run_radon_raw(self, code: str, question_id: str = "", question_title: str = "",
                      difficulty: str = "", code_index: int = 0) -> Dict:
        """
        Run Radon raw metrics analysis

        Args:
            code: Python code
            question_id: Question ID (for recording)
            question_title: Question title (for recording)
            difficulty: Difficulty (for recording)
            code_index: Code index (for recording)

        Returns:
            Raw metrics dictionary
        """
        # First extract clean code
        code = self.extract_code(code)

        if not code or code.strip() == "":
            return {'loc': 0, 'lloc': 0, 'sloc': 0, 'comments': 0, 'blank': 0}

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_file = f.name

        try:
            # Run radon raw (raw metrics)
            result = subprocess.run(
                ['radon', 'raw', '-s', '-j', temp_file],
                capture_output=True,
                text=True,
                timeout=10
            )

            # Save raw output and add to buffer (if question info provided)
            if question_id and hasattr(self, '_current_raw_entry'):
                self._current_raw_entry['raw_output'] = result.stdout if result.stdout else "No output"
                # Complete entry, add to buffer
                full_entry = {
                    'question_id': question_id,
                    'question_title': question_title,
                    'difficulty': difficulty,
                    'code_index': code_index,
                    'timestamp': self._current_raw_entry.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                    'cc_output': self._current_raw_entry.get('cc_output', 'No output'),
                    'mi_output': self._current_raw_entry.get('mi_output', 'No output'),
                    'raw_output': self._current_raw_entry.get('raw_output', 'No output')
                }
                self.raw_data_buffer.append(full_entry)
                # Clear temporary entry
                delattr(self, '_current_raw_entry')

            # Parse JSON output
            try:
                data = json.loads(result.stdout)
                if temp_file in data:
                    raw_data = data[temp_file]
                    return {
                        'loc': raw_data.get('loc', 0),
                        'lloc': raw_data.get('lloc', 0),
                        'sloc': raw_data.get('sloc', 0),
                        'comments': raw_data.get('comments', 0),
                        'blank': raw_data.get('blank', 0)
                    }
            except (json.JSONDecodeError, KeyError):
                pass

            return {'loc': 0, 'lloc': 0, 'sloc': 0, 'comments': 0, 'blank': 0}

        except subprocess.TimeoutExpired:
            return {'loc': 0, 'lloc': 0, 'sloc': 0, 'comments': 0, 'blank': 0}
        except Exception as e:
            return {'loc': 0, 'lloc': 0, 'sloc': 0, 'comments': 0, 'blank': 0}
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def get_worst_grade(self, grades: List[str]) -> str:
        """Get worst complexity grade"""
        grade_order = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}
        if not grades:
            return 'A'
        worst = max(grades, key=lambda g: grade_order.get(g, 0))
        return worst

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

            # Check if it's list format
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

    def save_detailed_results(self, llm_name: str, detailed_results: List[Dict],
                              output_dir: str = "radon_result", difficulty_filter: str = None,
                              correctness_filter: bool = None):
        """
        Save detailed radon results to separate CSV file

        Args:
            llm_name: LLM name
            detailed_results: Detailed results list
            output_dir: Output directory
            difficulty_filter: Difficulty filter (None, 'easy', 'medium', 'hard')
            correctness_filter: Correctness filter (None, True=correct, False=incorrect)
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Determine filename based on filter conditions
        filename_parts = [llm_name]

        if difficulty_filter:
            filename_parts.append(difficulty_filter.capitalize())

        if correctness_filter is not None:
            filename_parts.append("Correct" if correctness_filter else "Incorrect")

        filename = "_".join(filename_parts) + "_Code_Radon.csv"
        output_file = os.path.join(output_dir, filename)

        # Filter results
        filtered_results = detailed_results

        if difficulty_filter:
            filtered_results = [r for r in filtered_results if r['difficulty'].lower() == difficulty_filter.lower()]

        if correctness_filter is not None:
            filtered_results = [r for r in filtered_results if r['passed_tests'] == correctness_filter]

        if not filtered_results:
            filter_desc = []
            if difficulty_filter:
                filter_desc.append(f"{difficulty_filter} difficulty")
            if correctness_filter is not None:
                filter_desc.append("correct" if correctness_filter else "incorrect")
            print(f"  Warning: no {' + '.join(filter_desc)} codes")
            return

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow([
                'Question ID',
                'Question Title',
                'Difficulty',
                'Code Index',
                'Correctness',
                'Passed Tests',
                'Question Pass@1',
                'Complexity',
                'Avg Complexity',
                'Complexity Grade',
                'Functions',
                'Maintainability Index',
                'LOC',
                'LLOC',
                'SLOC',
                'Comments',
                'Blank Lines',
                'Source Code'
            ])

            # Write detailed results for each code
            for result in filtered_results:
                # Get source code
                source_code = result.get('source_code', '')

                # Determine correctness value
                correctness = 'Correct' if result['passed_tests'] else 'Incorrect'

                writer.writerow([
                    result['question_id'],
                    result['question_title'],
                    result['difficulty'],
                    result['code_index'],
                    correctness,
                    'Yes' if result['passed_tests'] else 'No',
                    result['question_pass@1'],
                    result['complexity'],
                    result['avg_complexity'],
                    result['grade'],
                    result['functions'],
                    result['maintainability'],
                    result['loc'],
                    result['lloc'],
                    result['sloc'],
                    result['comments'],
                    result['blank'],
                    source_code
                ])

        filter_desc = []
        if difficulty_filter:
            filter_desc.append(f"({difficulty_filter})")
        if correctness_filter is not None:
            filter_desc.append(f"({'Correct' if correctness_filter else 'Incorrect'})")
        print(f"  Detailed results{''.join(filter_desc)} saved to: {output_file}")

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
                    # Extract base LLM name (remove all markers)
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

        # Multi-dimensional statistics structure: difficulty x correctness
        multi_stats = {}

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

                for idx, code in enumerate(code_list):
                    if code and code.strip():
                        # Get correctness of current code
                        is_correct = graded_list[idx] if idx < len(graded_list) else False
                        correctness_key = 'correct' if is_correct else 'incorrect'

                        # Initialize statistics structure
                        for dim_key in ['all', difficulty, correctness_key, f"{difficulty}_{correctness_key}"]:
                            if dim_key not in multi_stats:
                                multi_stats[dim_key] = {
                                    'total_complexity': 0,
                                    'total_mi': 0.0,
                                    'total_loc': 0,
                                    'total_lloc': 0,
                                    'total_sloc': 0,
                                    'total_comments': 0,
                                    'total_blank': 0,
                                    'total_functions': 0,
                                    'valid_codes': 0,
                                    'grade_counts': {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0}
                                }

                        print(f"    Analyzing {question_id} code [{idx + 1}/{len(code_list)}]...", end='\r')

                        # Cyclomatic complexity analysis
                        cc_result = self.run_radon_cc(code, question_id, question_title, difficulty, idx + 1)
                        # Maintainability index
                        mi = self.run_radon_mi(code, question_id)
                        # Raw metrics
                        raw_result = self.run_radon_raw(code, question_id, question_title, difficulty, idx + 1)

                        # Update multi-dimensional statistics
                        for dim_key in ['all', difficulty, correctness_key, f"{difficulty}_{correctness_key}"]:
                            stats = multi_stats[dim_key]
                            stats['total_complexity'] += cc_result.get('complexity', 0)
                            stats['total_functions'] += cc_result.get('functions', 0)
                            grade = cc_result.get('grade', 'A')
                            if grade in stats['grade_counts']:
                                stats['grade_counts'][grade] += 1
                            stats['total_mi'] += mi
                            stats['total_loc'] += raw_result.get('loc', 0)
                            stats['total_lloc'] += raw_result.get('lloc', 0)
                            stats['total_sloc'] += raw_result.get('sloc', 0)
                            stats['total_comments'] += raw_result.get('comments', 0)
                            stats['total_blank'] += raw_result.get('blank', 0)
                            stats['valid_codes'] += 1

                        # Save detailed result
                        detailed_result = {
                            'question_id': question_id,
                            'question_title': question_title,
                            'difficulty': difficulty,
                            'code_index': idx + 1,
                            'passed_tests': is_correct,
                            'question_pass@1': pass_at_1,
                            'complexity': cc_result.get('complexity', 0),
                            'avg_complexity': cc_result.get('avg_complexity', 0),
                            'grade': grade,
                            'functions': cc_result.get('functions', 0),
                            'maintainability': round(mi, 2),
                            'loc': raw_result.get('loc', 0),
                            'lloc': raw_result.get('lloc', 0),
                            'sloc': raw_result.get('sloc', 0),
                            'comments': raw_result.get('comments', 0),
                            'blank': raw_result.get('blank', 0),
                            'source_code': code
                        }
                        all_detailed_results.append(detailed_result)

        print(f"\n  Analysis completed: {len(all_detailed_results)} valid codes")

        # Save raw radon output data
        self.save_raw_data(llm_name, output_dir="radon_raw")

        # Save detailed results for all codes (overall)
        self.save_detailed_results(llm_name, all_detailed_results, output_dir="radon_result")

        # Save detailed results by difficulty
        for difficulty in ['easy', 'medium', 'hard']:
            self.save_detailed_results(llm_name, all_detailed_results,
                                       output_dir="radon_result",
                                       difficulty_filter=difficulty)

        # Save detailed results by correctness
        for correctness in [True, False]:
            self.save_detailed_results(llm_name, all_detailed_results,
                                       output_dir="radon_result",
                                       correctness_filter=correctness)

        # Save detailed results by difficulty + correctness
        for difficulty in ['easy', 'medium', 'hard']:
            for correctness in [True, False]:
                self.save_detailed_results(llm_name, all_detailed_results,
                                           output_dir="radon_result",
                                           difficulty_filter=difficulty,
                                           correctness_filter=correctness)

        # Generate summary results
        results = {}

        # Generate statistical results for all dimensions
        for key, stats in multi_stats.items():
            valid_codes = stats['valid_codes']
            if valid_codes == 0:
                continue

            avg_complexity = stats['total_complexity'] / valid_codes
            avg_mi = stats['total_mi'] / valid_codes
            avg_loc = stats['total_loc'] / valid_codes

            # Filter results for this dimension to calculate correctness rate
            if key == 'all':
                dimension_results = all_detailed_results
                model_label = llm_name
            elif key in ['easy', 'medium', 'hard']:
                dimension_results = [r for r in all_detailed_results if r['difficulty'] == key]
                model_label = f"{llm_name} ({key.capitalize()})"
            elif key in ['correct', 'incorrect']:
                is_correct = (key == 'correct')
                dimension_results = [r for r in all_detailed_results if r['passed_tests'] == is_correct]
                model_label = f"{llm_name} ({'Correct' if is_correct else 'Incorrect'})"
            else:  # difficulty_correctness
                parts = key.split('_')
                difficulty = parts[0]
                is_correct = (parts[1] == 'correct')
                dimension_results = [r for r in all_detailed_results
                                     if r['difficulty'] == difficulty and r['passed_tests'] == is_correct]
                model_label = f"{llm_name} ({difficulty.capitalize()}-{'Correct' if is_correct else 'Incorrect'})"

            correctness_rate = self.calculate_correctness_rate(dimension_results)

            results[key] = {
                'model': model_label,
                'total_codes': valid_codes,
                'correctness_rate': correctness_rate,
                'avg_complexity': round(avg_complexity, 2),
                'total_complexity': stats['total_complexity'],
                'avg_maintainability': round(avg_mi, 2),
                'avg_loc': round(avg_loc, 2),
                'total_loc': stats['total_loc'],
                'total_lloc': stats['total_lloc'],
                'total_sloc': stats['total_sloc'],
                'total_comments': stats['total_comments'],
                'total_blank': stats['total_blank'],
                'total_functions': stats['total_functions'],
                'grade_A': stats['grade_counts']['A'],
                'grade_B': stats['grade_counts']['B'],
                'grade_C': stats['grade_counts']['C'],
                'grade_D': stats['grade_counts']['D'],
                'grade_E': stats['grade_counts']['E'],
                'grade_F': stats['grade_counts']['F']
            }

        # Print statistics
        print(f"\n  Overall statistics:")
        if 'all' in results:
            r = results['all']
            print(f"    Code count: {r['total_codes']}")
            print(f"    Correctness rate: {r['correctness_rate']}%")
            print(f"    Average cyclomatic complexity: {r['avg_complexity']}")
            print(f"    Average maintainability index: {r['avg_maintainability']}")
            print(f"    Average lines of code: {r['avg_loc']}")

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
                    'Avg Complexity',
                    'Total Complexity',
                    'Avg Maintainability Index',
                    'Avg LOC',
                    'Total LOC',
                    'Total LLOC',
                    'Total SLOC',
                    'Total Comments',
                    'Total Blank',
                    'Total Functions',
                    'Grade A',
                    'Grade B',
                    'Grade C',
                    'Grade D',
                    'Grade E',
                    'Grade F'
                ])

            # Write data
            for result in self.results:
                writer.writerow([
                    result['model'],
                    result['total_codes'],
                    result['correctness_rate'],
                    result['avg_complexity'],
                    result['total_complexity'],
                    result['avg_maintainability'],
                    result['avg_loc'],
                    result['total_loc'],
                    result['total_lloc'],
                    result['total_sloc'],
                    result['total_comments'],
                    result['total_blank'],
                    result['total_functions'],
                    result['grade_A'],
                    result['grade_B'],
                    result['grade_C'],
                    result['grade_D'],
                    result['grade_E'],
                    result['grade_F']
                ])

        if not append:
            print(f"\nSummary results saved to: {output_file}")
        else:
            print(f"\nSummary results appended to: {output_file}")

    def run_batch_analysis(self, submission_dir="submissions", summary_output="radon_result/radon_summary.csv"):
        """
        Batch analyze all LLM results in the submissions folder

        Args:
            submission_dir: Submissions folder path
            summary_output: Summary results output file
        """
        # Check if radon is installed
        if not self.check_radon_installed():
            print("Error: Radon is not installed")
            print("Please run: pip install radon")
            return

        print("Radon is installed, starting code complexity analysis...")

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
        print("Starting batch code complexity analysis")
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

                # Define output order: overall -> by difficulty -> by correctness -> by difficulty+correctness
                output_order = [
                    'all',
                    'easy', 'medium', 'hard',
                    'correct', 'incorrect',
                    'easy_correct', 'easy_incorrect',
                    'medium_correct', 'medium_incorrect',
                    'hard_correct', 'hard_incorrect'
                ]

                # Add results in order
                for key in output_order:
                    if key in results:
                        self.results.append(results[key])

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
                          f"average complexity={result['avg_complexity']}, "
                          f"average maintainability={result['avg_maintainability']}")

        # Final summary
        print("\n" + "=" * 60)
        print("Code complexity analysis completed!")
        print("=" * 60)
        print(f"\nAll results saved to: {summary_output}")
        print(f"Processed this run: {len(llm_folders_to_process)} LLMs")
        print(f"Skipped (completed): {len(skipped_llms)} LLMs")
        print(f"Total: {len(llm_folders)} LLMs")


if __name__ == "__main__":
    checker = RadonChecker()

    # Run batch analysis
    checker.run_batch_analysis(
        submission_dir="submissions",  # Submissions folder path
        summary_output="radon_result/radon_summary.csv"  # Summary results filename
    )

    print("\nCompleted!")
    print("\nGenerated files:")
    print(
        "  - radon_result/radon_summary.csv: Summary results for all LLMs (including overall, by difficulty, by correctness, by difficulty+correctness)")
    print("  - radon_result/<LLM>_Code_Radon.csv: Detailed evaluation for all codes of each LLM")
    print("  - radon_result/<LLM>_Easy_Code_Radon.csv: Detailed evaluation for Easy difficulty codes of each LLM")
    print("  - radon_result/<LLM>_Medium_Code_Radon.csv: Detailed evaluation for Medium difficulty codes of each LLM")
    print("  - radon_result/<LLM>_Hard_Code_Radon.csv: Detailed evaluation for Hard difficulty codes of each LLM")
    print("  - radon_result/<LLM>_Correct_Code_Radon.csv: Detailed evaluation for correct codes of each LLM")
    print("  - radon_result/<LLM>_Incorrect_Code_Radon.csv: Detailed evaluation for incorrect codes of each LLM")
    print("  - radon_result/<LLM>_Easy_Correct_Code_Radon.csv: Easy+Correct combination")
    print("  - radon_result/<LLM>_Easy_Incorrect_Code_Radon.csv: Easy+Incorrect combination")
    print("  - radon_result/<LLM>_Medium_Correct_Code_Radon.csv: Medium+Correct combination")
    print("  - radon_result/<LLM>_Medium_Incorrect_Code_Radon.csv: Medium+Incorrect combination")
    print("  - radon_result/<LLM>_Hard_Correct_Code_Radon.csv: Hard+Correct combination")
    print("  - radon_result/<LLM>_Hard_Incorrect_Code_Radon.csv: Hard+Incorrect combination")
    print("  - radon_raw/<LLM>/<LLM>_raw.txt: Raw radon output data for each LLM")