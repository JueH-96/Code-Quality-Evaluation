#!/usr/bin/env python3
import json
import os
import csv
import subprocess
import glob
import time
import urllib.parse
from typing import Dict, List, Tuple
from collections import defaultdict


class SonarQubeChecker:
    def __init__(self):
        """Initialize SonarQube checker"""
        self.results = []
        self.question_difficulty_map = {}
        self.llm_code_stats = {}
        self.fetch_errors = []
        self.question_pass_status = {}  # New: Store pass status for each question {llm_name: {question_id: 'Yes'/'No'}}

    def sanitize_filename(self, filename: str) -> str:
        """Clean filename, remove or replace illegal characters"""
        illegal_chars = '<>:"/\\|?*'
        for char in illegal_chars:
            filename = filename.replace(char, '_')
        filename = filename.strip()
        if not filename:
            filename = "untitled"
        return filename

    def save_raw_data(self, llm_name: str, raw_data: str, page: int = None):
        """Save curl raw data to file"""
        raw_dir = os.path.join("sonarqube_raw", llm_name)
        os.makedirs(raw_dir, exist_ok=True)

        safe_llm_name = self.sanitize_filename(llm_name)

        if page is not None:
            filename = f"{safe_llm_name}_page{page}.txt"
        else:
            filename = f"{safe_llm_name}.txt"

        filepath = os.path.join(raw_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(raw_data)

    def load_code_stats_from_json(self, submission_dir: str = "submissions"):
        """Load code statistics (count and correctness rate) for each LLM from JSON files, and pass status"""
        print("Loading code statistics from JSON files...")

        llm_folders = [
            d for d in os.listdir(submission_dir)
            if os.path.isdir(os.path.join(submission_dir, d)) and not d.startswith('.')
        ]

        total_llms = 0
        total_codes = 0

        for llm_name in llm_folders:
            llm_folder = os.path.join(submission_dir, llm_name)
            self.llm_code_stats[llm_name] = {}
            self.question_difficulty_map[llm_name] = {}
            self.question_pass_status[llm_name] = {}  # Initialize pass status dictionary

            json_files = glob.glob(os.path.join(llm_folder, "*.json"))
            jsonl_files = glob.glob(os.path.join(llm_folder, "*.jsonl"))
            all_files = json_files + jsonl_files

            for json_file in all_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        try:
                            data = json.load(f)
                        except json.JSONDecodeError:
                            f.seek(0)
                            data = []
                            for line in f:
                                if line.strip():
                                    data.append(json.loads(line))

                        if isinstance(data, list):
                            for item in data:
                                question_id = str(item.get('question_id', '')).strip()
                                difficulty = item.get('difficulty', 'unknown').lower()
                                graded_list = item.get('graded_list', [])

                                if question_id:
                                    # Save difficulty mapping
                                    self.question_difficulty_map[llm_name][question_id] = difficulty

                                    # Determine if code passed all tests
                                    # If all tests in graded_list are True, then Yes, otherwise No
                                    if graded_list and all(grade is True for grade in graded_list):
                                        self.question_pass_status[llm_name][question_id] = 'Yes'
                                    else:
                                        self.question_pass_status[llm_name][question_id] = 'No'

                                    # Count code statistics and correctness rate
                                    if difficulty not in self.llm_code_stats[llm_name]:
                                        self.llm_code_stats[llm_name][difficulty] = {
                                            'total': 0,
                                            'correct': 0
                                        }

                                    # Calculate number of correct codes (at least one test passed)
                                    correct_count = sum(1 for grade in graded_list if grade is True)

                                    self.llm_code_stats[llm_name][difficulty]['total'] += 1
                                    if correct_count > 0:
                                        self.llm_code_stats[llm_name][difficulty]['correct'] += 1

                                    total_codes += 1

                except Exception as e:
                    print(f"  Warning: Error processing file {json_file}: {e}")
                    continue

            if llm_name in self.llm_code_stats and self.llm_code_stats[llm_name]:
                total_llms += 1

        print(f"  Loaded statistics for {total_llms} LLMs")
        print(f"  Total {total_codes} code files")

        # Print statistics for each LLM
        for llm_name, stats in self.llm_code_stats.items():
            total = sum(d['total'] for d in stats.values())
            correct = sum(d['correct'] for d in stats.values())
            rate = (correct / total * 100) if total > 0 else 0
            print(f"    {llm_name}: {total} codes, {correct} correct ({rate:.2f}%)")

    def get_difficulty_from_filename(self, filename: str, llm_name: str) -> str:
        """Find corresponding difficulty from filename"""
        name = filename.replace('.py', '').lstrip('qQ')

        if llm_name in self.question_difficulty_map:
            difficulty = self.question_difficulty_map[llm_name].get(name, None)
            if difficulty:
                return difficulty

        for llm, mapping in self.question_difficulty_map.items():
            if name in mapping:
                return mapping[name]

        return 'unknown'

    def get_pass_status_from_filename(self, filename: str, llm_name: str) -> str:
        """Find corresponding pass status from filename"""
        name = filename.replace('.py', '').lstrip('qQ')

        if llm_name in self.question_pass_status:
            status = self.question_pass_status[llm_name].get(name, None)
            if status:
                return status

        # If not found in current LLM, try to find in other LLMs
        for llm, mapping in self.question_pass_status.items():
            if name in mapping:
                return mapping[name]

        return 'Unknown'

    def get_code_stats(self, llm_name: str, difficulty: str = None) -> Tuple[int, float]:
        """Get code statistics for specified LLM (total count and correctness rate)"""
        if llm_name not in self.llm_code_stats:
            return 0, 0.0

        if difficulty:
            if difficulty not in self.llm_code_stats[llm_name]:
                return 0, 0.0

            stats = self.llm_code_stats[llm_name][difficulty]
            total = stats['total']
            correct = stats['correct']
            rate = (correct / total * 100) if total > 0 else 0.0
            return total, rate
        else:
            total = sum(d['total'] for d in self.llm_code_stats[llm_name].values())
            correct = sum(d['correct'] for d in self.llm_code_stats[llm_name].values())
            rate = (correct / total * 100) if total > 0 else 0.0
            return total, rate

    def fetch_all_llm_folders(self, organization_key: str, component_key: str, auth_token: str) -> List[str]:
        """Get all LLM folder names in the project"""
        print(f"Fetching LLM folder list...")

        url = f"https://sonarcloud.io/api/components/tree?component={component_key}&qualifiers=DIR&ps=500"

        try:
            result = subprocess.run(
                ['curl', '-s', '-u', f'{auth_token}:', url],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                response = json.loads(result.stdout)
                components = response.get('components', [])

                llm_folders = []
                for comp in components:
                    key = comp.get('key', '')
                    path = comp.get('path', '')
                    if path and '/' not in path:
                        llm_folders.append(path)

                print(f"  Found {len(llm_folders)} LLM folders")
                return llm_folders
        except Exception as e:
            print(f"  Warning: Unable to fetch LLM folder list: {e}")

        return []

    def verify_llm_component_exists(self, organization_key: str, component_key: str,
                                    auth_token: str, llm_name: str) -> bool:
        """Verify if LLM component exists in SonarQube"""
        llm_component = f"{component_key}:{llm_name}"
        llm_component_encoded = urllib.parse.quote(llm_component, safe='')

        url = f"https://sonarcloud.io/api/components/show?component={llm_component_encoded}"

        try:
            result = subprocess.run(
                ['curl', '-s', '-u', f'{auth_token}:', url],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                response = json.loads(result.stdout)

                debug_file = f"sonarqube_raw/debug_{self.sanitize_filename(llm_name)}_verify.txt"
                os.makedirs("sonarqube_raw", exist_ok=True)
                with open(debug_file, 'w', encoding='utf-8') as f:
                    f.write(f"LLM Name: {llm_name}\n")
                    f.write(f"Component: {llm_component}\n")
                    f.write(f"Encoded: {llm_component_encoded}\n")
                    f.write(f"URL: {url}\n")
                    f.write(f"Response: {json.dumps(response, indent=2, ensure_ascii=False)}\n")

                if 'component' in response:
                    print(f"      ✓ '{llm_name}' component exists")
                    return True
                elif 'errors' in response:
                    print(f"      ✗ '{llm_name}' API error: {response['errors']}")
                    return False
        except Exception as e:
            print(f"      ✗ '{llm_name}' verification failed: {e}")
            return False

        print(f"      ✗ '{llm_name}' component does not exist")
        return False

    def fetch_sonarqube_issues_by_llm(self, organization_key: str, component_key: str,
                                      auth_token: str, llm_name: str,
                                      max_retries: int = 3) -> List[Dict]:
        """Fetch all issues for specified LLM (with retry mechanism, no page limit)"""
        all_issues = []
        page = 1
        page_size = 500
        consecutive_failures = 0
        max_consecutive_failures = 3

        llm_component = f"{component_key}:{llm_name}"
        llm_component_encoded = urllib.parse.quote(llm_component, safe='')

        print(f"    Starting to fetch issues for {llm_name}")
        print(f"    Component key: {llm_component}")
        print(f"    URL encoded: {llm_component_encoded}")

        while True:
            url = f"https://sonarcloud.io/api/issues/search?organization={organization_key}&componentKeys={llm_component_encoded}&ps={page_size}&p={page}"

            success = False
            response_data = None

            for retry in range(max_retries):
                try:
                    result = subprocess.run(
                        ['curl', '-s', '-u', f'{auth_token}:', url],
                        capture_output=True,
                        text=True,
                        timeout=60
                    )

                    if result.returncode != 0:
                        print(f"      ⚠️  Page {page} curl return code: {result.returncode}")
                        print(f"      Debug URL: {url}")
                        if retry < max_retries - 1:
                            time.sleep(2 ** retry)
                            continue
                        else:
                            error_msg = f"LLM: {llm_name}, Page: {page}, Error: curl failed with code {result.returncode}, URL: {url}"
                            self.fetch_errors.append(error_msg)
                            print(f"      ✖ {error_msg}")
                            break

                    self.save_raw_data(llm_name, result.stdout, page)

                    try:
                        response_data = json.loads(result.stdout)
                    except json.JSONDecodeError as je:
                        print(f"      ⚠️  Page {page} JSON parsing failed: {je}")
                        self.save_raw_data(llm_name, f"PARSE_ERROR_{result.stdout}", page)
                        if retry < max_retries - 1:
                            time.sleep(2 ** retry)
                            continue
                        else:
                            error_msg = f"LLM: {llm_name}, Page: {page}, Error: JSON decode failed"
                            self.fetch_errors.append(error_msg)
                            print(f"      ✖ {error_msg}")
                            break

                    if 'errors' in response_data:
                        error_msg = f"LLM: {llm_name}, Page: {page}, API Error: {response_data['errors']}"
                        self.fetch_errors.append(error_msg)
                        print(f"      ✖ {error_msg}")
                        break

                    issues = response_data.get('issues', [])
                    paging = response_data.get('paging', {})

                    if not issues and page == 1:
                        print(f"      ℹ️  {llm_name} has no issues")
                        success = True
                        break

                    if not issues and page > 1:
                        print(f"      ✓ All data fetched")
                        success = True
                        break

                    all_issues.extend(issues)

                    total = paging.get('total', 0)
                    current_count = len(all_issues)

                    print(f"      ✓ Page {page}: Fetched {len(issues)} issues (cumulative: {current_count}/{total})")

                    if total > 0 and current_count >= total:
                        print(f"      ✓ All {total} issues fetched")
                        success = True
                        break

                    if len(issues) < page_size:
                        print(f"      ✓ Reached last page (returned {len(issues)} < {page_size})")
                        success = True
                        break

                    success = True
                    break

                except subprocess.TimeoutExpired:
                    print(f"      ⚠️  Page {page} timeout (attempt {retry + 1}/{max_retries})")
                    if retry < max_retries - 1:
                        time.sleep(2 ** retry)
                        continue
                    else:
                        error_msg = f"LLM: {llm_name}, Page: {page}, Error: Timeout"
                        self.fetch_errors.append(error_msg)
                        print(f"      ✖ {error_msg}")
                        break

                except Exception as e:
                    print(
                        f"      ⚠️  Page {page} exception: {type(e).__name__}: {e} (attempt {retry + 1}/{max_retries})")
                    if retry < max_retries - 1:
                        time.sleep(2 ** retry)
                        continue
                    else:
                        error_msg = f"LLM: {llm_name}, Page: {page}, Error: {type(e).__name__}: {str(e)}"
                        self.fetch_errors.append(error_msg)
                        print(f"      ✖ {error_msg}")
                        break

            if not success:
                consecutive_failures += 1
                print(f"      ⚠️  Consecutive failures: {consecutive_failures}/{max_consecutive_failures}")

                if consecutive_failures >= max_consecutive_failures:
                    error_msg = f"LLM: {llm_name}, {consecutive_failures} consecutive page failures, stopping fetch"
                    self.fetch_errors.append(error_msg)
                    print(f"      ✖ {error_msg}")
                    break
            else:
                consecutive_failures = 0

                if response_data:
                    issues = response_data.get('issues', [])
                    paging = response_data.get('paging', {})
                    total = paging.get('total', 0)

                    if (not issues) or \
                            (total > 0 and len(all_issues) >= total) or \
                            (len(issues) < page_size):
                        break

            page += 1
            time.sleep(0.5)

        print(f"    ✅ Completed: {llm_name} fetched {len(all_issues)} issues total ({page} pages)")

        if response_data and 'paging' in response_data:
            expected_total = response_data['paging'].get('total', 0)
            if expected_total > 0 and len(all_issues) < expected_total:
                warning_msg = f"⚠️  {llm_name} data may be incomplete: fetched {len(all_issues)}/{expected_total}"
                print(f"    {warning_msg}")
                self.fetch_errors.append(f"LLM: {llm_name}, Warning: {warning_msg}")

        return all_issues

    def fetch_sonarqube_issues(self, organization_key: str, component_key: str, auth_token: str) -> List[Dict]:
        """Fetch issue list from SonarQube API (query by LLM separately)"""
        print(f"Fetching SonarQube issues...")
        print(f"  Organization: {organization_key}")
        print(f"  Component: {component_key}")

        llm_folders = self.fetch_all_llm_folders(organization_key, component_key, auth_token)

        all_issues = []

        if llm_folders:
            print(f"\nQuerying {len(llm_folders)} LLMs separately")

            for idx, llm_name in enumerate(sorted(llm_folders), 1):
                print(f"\n  [{idx}/{len(llm_folders)}] Querying {llm_name}...")

                llm_issues = self.fetch_sonarqube_issues_by_llm(
                    organization_key, component_key, auth_token, llm_name
                )

                print(f"    Fetched {len(llm_issues)} issues")
                all_issues.extend(llm_issues)

                print(f"    Cumulative: {len(all_issues)} issues")

        print(f"\n  ✓ Total fetched {len(all_issues)} issues")

        return all_issues

    def parse_issues_by_llm_and_file(self, issues: List[Dict]) -> Dict[str, Dict[str, List[Dict]]]:
        """Group issues by LLM and filename"""
        llm_file_issues = defaultdict(lambda: defaultdict(list))

        for issue in issues:
            component = issue.get('component', '')
            if ':' in component:
                file_path = component.split(':', 1)[1]

                if '/' in file_path:
                    parts = file_path.split('/', 1)
                    llm_name = parts[0]
                    filename = parts[1] if len(parts) > 1 else file_path
                else:
                    llm_name = 'root'
                    filename = file_path

                llm_file_issues[llm_name][filename].append(issue)
            else:
                if '/' in component:
                    parts = component.split('/', 1)
                    llm_name = parts[0]
                    filename = parts[1] if len(parts) > 1 else component
                else:
                    llm_name = 'root'
                    filename = component

                llm_file_issues[llm_name][filename].append(issue)

        return dict(llm_file_issues)

    def create_empty_results(self, llm_name: str) -> Dict:
        """Create empty results (when LLM has no issues)"""
        empty_stats = {
            'total_issues': 0,
            'severity_blocker': 0, 'severity_critical': 0, 'severity_major': 0,
            'severity_minor': 0, 'severity_info': 0,
            'type_bug': 0, 'type_vulnerability': 0, 'type_code_smell': 0,
            'type_security_hotspot': 0,
            'attr_consistency': 0, 'attr_intentionality': 0,
            'attr_adaptability': 0, 'attr_responsibility': 0,
            'sq_security_blocker': 0, 'sq_security_high': 0, 'sq_security_medium': 0,
            'sq_security_low': 0, 'sq_security_info': 0,
            'sq_reliability_blocker': 0, 'sq_reliability_high': 0, 'sq_reliability_medium': 0,
            'sq_reliability_low': 0, 'sq_reliability_info': 0,
            'sq_maintainability_blocker': 0, 'sq_maintainability_high': 0, 'sq_maintainability_medium': 0,
            'sq_maintainability_low': 0, 'sq_maintainability_info': 0
        }

        total_codes, correctness_rate = self.get_code_stats(llm_name)

        results = {
            'all': {
                'model': llm_name,
                'total_codes': total_codes,
                'correctness_rate': correctness_rate,
                'avg_issues_per_code': 0
            }
        }
        results['all'].update(empty_stats)

        if llm_name in self.llm_code_stats:
            for difficulty in self.llm_code_stats[llm_name].keys():
                diff_codes, diff_rate = self.get_code_stats(llm_name, difficulty)
                results[difficulty] = {
                    'model': f"{llm_name} ({difficulty})",
                    'total_codes': diff_codes,
                    'correctness_rate': diff_rate,
                    'avg_issues_per_code': 0
                }
                results[difficulty].update(empty_stats)

        return results

    def analyze_llm_issues(self, llm_name: str, file_issues: Dict[str, List[Dict]]) -> Dict:
        """Analyze SonarQube issues for a single LLM"""
        print(f"\nAnalyzing LLM: {llm_name}")

        total_codes_from_json, overall_correctness = self.get_code_stats(llm_name)
        print(f"  JSON statistics: {total_codes_from_json} codes, correctness rate: {overall_correctness:.2f}%")

        if not file_issues or len(file_issues) == 0:
            print(f"  No issues found for this LLM")
            return self.create_empty_results(llm_name)

        print(f"  Involves {len(file_issues)} files with issues")

        all_detailed_results = []
        difficulty_stats = {}

        for filename, file_issue_list in file_issues.items():
            difficulty = self.get_difficulty_from_filename(filename, llm_name)
            pass_status = self.get_pass_status_from_filename(filename, llm_name)  # Get pass status

            if difficulty not in difficulty_stats:
                difficulty_stats[difficulty] = {
                    'total_issues': 0,
                    'files_with_issues': 0,
                    'severity_blocker': 0, 'severity_critical': 0, 'severity_major': 0,
                    'severity_minor': 0, 'severity_info': 0,
                    'type_bug': 0, 'type_vulnerability': 0, 'type_code_smell': 0,
                    'type_security_hotspot': 0,
                    'attr_consistency': 0, 'attr_intentionality': 0,
                    'attr_adaptability': 0, 'attr_responsibility': 0,
                    'sq_security_blocker': 0, 'sq_security_high': 0, 'sq_security_medium': 0,
                    'sq_security_low': 0, 'sq_security_info': 0,
                    'sq_reliability_blocker': 0, 'sq_reliability_high': 0, 'sq_reliability_medium': 0,
                    'sq_reliability_low': 0, 'sq_reliability_info': 0,
                    'sq_maintainability_blocker': 0, 'sq_maintainability_high': 0, 'sq_maintainability_medium': 0,
                    'sq_maintainability_low': 0, 'sq_maintainability_info': 0
                }

            file_stats = {
                'severity_blocker': 0, 'severity_critical': 0, 'severity_major': 0,
                'severity_minor': 0, 'severity_info': 0,
                'type_bug': 0, 'type_vulnerability': 0, 'type_code_smell': 0,
                'type_security_hotspot': 0,
                'attr_consistency': 0, 'attr_intentionality': 0,
                'attr_adaptability': 0, 'attr_responsibility': 0,
                'sq_security_blocker': 0, 'sq_security_high': 0, 'sq_security_medium': 0,
                'sq_security_low': 0, 'sq_security_info': 0,
                'sq_reliability_blocker': 0, 'sq_reliability_high': 0, 'sq_reliability_medium': 0,
                'sq_reliability_low': 0, 'sq_reliability_info': 0,
                'sq_maintainability_blocker': 0, 'sq_maintainability_high': 0, 'sq_maintainability_medium': 0,
                'sq_maintainability_low': 0, 'sq_maintainability_info': 0
            }

            for issue in file_issue_list:
                severity = issue.get('severity', '').upper()
                if severity == 'BLOCKER':
                    file_stats['severity_blocker'] += 1
                elif severity == 'CRITICAL':
                    file_stats['severity_critical'] += 1
                elif severity == 'MAJOR':
                    file_stats['severity_major'] += 1
                elif severity == 'MINOR':
                    file_stats['severity_minor'] += 1
                elif severity == 'INFO':
                    file_stats['severity_info'] += 1

                issue_type = issue.get('type', '').upper()
                if issue_type == 'BUG':
                    file_stats['type_bug'] += 1
                elif issue_type == 'VULNERABILITY':
                    file_stats['type_vulnerability'] += 1
                elif issue_type == 'CODE_SMELL':
                    file_stats['type_code_smell'] += 1
                elif issue_type == 'SECURITY_HOTSPOT':
                    file_stats['type_security_hotspot'] += 1

                clean_attr = issue.get('cleanCodeAttribute', '').upper()
                if clean_attr in ['CONSISTENCY', 'CONSISTENT']:
                    file_stats['attr_consistency'] += 1
                elif clean_attr in ['INTENTIONALITY', 'INTENTIONAL']:
                    file_stats['attr_intentionality'] += 1
                elif clean_attr in ['ADAPTABILITY', 'ADAPTABLE']:
                    file_stats['attr_adaptability'] += 1
                elif clean_attr in ['RESPONSIBILITY', 'RESPONSIBLE']:
                    file_stats['attr_responsibility'] += 1

                impacts = issue.get('impacts', [])
                for impact in impacts:
                    software_quality = impact.get('softwareQuality', '').upper()
                    impact_severity = impact.get('severity', '').upper()

                    key = f"sq_{software_quality.lower()}_{impact_severity.lower()}"
                    if key in file_stats:
                        file_stats[key] += 1

            stats = difficulty_stats[difficulty]
            stats['total_issues'] += len(file_issue_list)
            stats['files_with_issues'] += 1

            for key, value in file_stats.items():
                stats[key] += value

            detailed_result = {
                'filename': filename,
                'difficulty': difficulty,
                'pass_status': pass_status,  # Add pass status
                'total_issues': len(file_issue_list),
                'issues_detail': file_issue_list
            }
            detailed_result.update(file_stats)

            all_detailed_results.append(detailed_result)

        print(f"  Analysis completed: {len(all_detailed_results)} files")

        self.save_detailed_results(llm_name, all_detailed_results, output_dir="sonarqube_result")

        for difficulty in difficulty_stats.keys():
            self.save_detailed_results(llm_name, all_detailed_results,
                                       output_dir="sonarqube_result",
                                       difficulty_filter=difficulty)

        results = {}

        total_issues = sum(stats['total_issues'] for stats in difficulty_stats.values())
        avg_issues = total_issues / total_codes_from_json if total_codes_from_json > 0 else 0

        total_stats = {}
        for stat_key in difficulty_stats[list(difficulty_stats.keys())[0]].keys():
            if stat_key not in ['files_with_issues']:
                total_stats[stat_key] = sum(stats[stat_key] for stats in difficulty_stats.values())

        results['all'] = {
            'model': llm_name,
            'total_codes': total_codes_from_json,
            'correctness_rate': overall_correctness,
            'avg_issues_per_code': round(avg_issues, 2)
        }
        results['all'].update(total_stats)

        print(f"  Overall statistics:")
        print(f"    Total codes: {total_codes_from_json}")
        print(f"    Correctness rate: {overall_correctness:.2f}%")
        print(f"    Total issues: {total_issues}")
        print(f"    Average issues: {round(avg_issues, 2)}")

        for difficulty in ['easy', 'medium', 'hard', 'unknown']:
            diff_codes, diff_rate = self.get_code_stats(llm_name, difficulty)

            if diff_codes == 0:
                continue

            if difficulty in difficulty_stats:
                stats = difficulty_stats[difficulty]
                diff_issues = stats['total_issues']
                avg_issues = diff_issues / diff_codes
            else:
                stats = {
                    'total_issues': 0,
                    'severity_blocker': 0, 'severity_critical': 0, 'severity_major': 0,
                    'severity_minor': 0, 'severity_info': 0,
                    'type_bug': 0, 'type_vulnerability': 0, 'type_code_smell': 0,
                    'type_security_hotspot': 0,
                    'attr_consistency': 0, 'attr_intentionality': 0,
                    'attr_adaptability': 0, 'attr_responsibility': 0,
                    'sq_security_blocker': 0, 'sq_security_high': 0, 'sq_security_medium': 0,
                    'sq_security_low': 0, 'sq_security_info': 0,
                    'sq_reliability_blocker': 0, 'sq_reliability_high': 0, 'sq_reliability_medium': 0,
                    'sq_reliability_low': 0, 'sq_reliability_info': 0,
                    'sq_maintainability_blocker': 0, 'sq_maintainability_high': 0, 'sq_maintainability_medium': 0,
                    'sq_maintainability_low': 0, 'sq_maintainability_info': 0
                }
                diff_issues = 0
                avg_issues = 0

            results[difficulty] = {
                'model': f"{llm_name} ({difficulty})",
                'total_codes': diff_codes,
                'correctness_rate': diff_rate,
                'avg_issues_per_code': round(avg_issues, 2)
            }

            for key, value in stats.items():
                if key not in ['files_with_issues']:
                    results[difficulty][key] = value

            print(f"  {difficulty.capitalize()} difficulty statistics:")
            print(f"    Codes: {diff_codes}")
            print(f"    Correctness rate: {diff_rate:.2f}%")
            print(f"    Total issues: {diff_issues}")
            print(f"    Average issues: {round(avg_issues, 2)}")

        return results

    def save_detailed_results(self, llm_name: str, detailed_results: List[Dict],
                              output_dir: str = "sonarqube_result", difficulty_filter: str = None):
        """Save detailed SonarQube results for each code to separate CSV file"""
        os.makedirs(output_dir, exist_ok=True)

        if difficulty_filter:
            output_file = os.path.join(output_dir, f"{llm_name}_{difficulty_filter.capitalize()}_Code_SonarQube.csv")
            filtered_results = [r for r in detailed_results if r['difficulty'].lower() == difficulty_filter.lower()]
        else:
            output_file = os.path.join(output_dir, f"{llm_name}_Code_SonarQube.csv")
            filtered_results = detailed_results

        if not filtered_results:
            return

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            writer.writerow([
                'File Name', 'Difficulty', 'Passed Tests', 'Total Issues',
                'Severity: Blocker', 'Severity: Critical', 'Severity: Major', 'Severity: Minor', 'Severity: Info',
                'Type: Bug', 'Type: Vulnerability', 'Type: Code Smell', 'Type: Security Hotspot',
                'Attribute: Consistency', 'Attribute: Intentionality', 'Attribute: Adaptability',
                'Attribute: Responsibility',
                'SQ Security: Blocker', 'SQ Security: High', 'SQ Security: Medium', 'SQ Security: Low',
                'SQ Security: Info',
                'SQ Reliability: Blocker', 'SQ Reliability: High', 'SQ Reliability: Medium', 'SQ Reliability: Low',
                'SQ Reliability: Info',
                'SQ Maintainability: Blocker', 'SQ Maintainability: High', 'SQ Maintainability: Medium',
                'SQ Maintainability: Low', 'SQ Maintainability: Info',
                'All Messages'
            ])

            for result in filtered_results:
                all_messages = []
                for issue in result['issues_detail']:
                    rule = issue.get('rule', 'unknown')
                    message = issue.get('message', '')
                    line = issue.get('line', '')
                    msg_str = f"[{rule}] {message}"
                    if line:
                        msg_str += f" (line {line})"
                    all_messages.append(msg_str)

                all_messages_str = " || ".join(all_messages) if all_messages else "None"

                writer.writerow([
                    result['filename'], result['difficulty'], result.get('pass_status', 'Unknown'),
                    result['total_issues'],
                    result.get('severity_blocker', 0), result.get('severity_critical', 0),
                    result.get('severity_major', 0),
                    result.get('severity_minor', 0), result.get('severity_info', 0),
                    result.get('type_bug', 0), result.get('type_vulnerability', 0), result.get('type_code_smell', 0),
                    result.get('type_security_hotspot', 0),
                    result.get('attr_consistency', 0), result.get('attr_intentionality', 0),
                    result.get('attr_adaptability', 0), result.get('attr_responsibility', 0),
                    result.get('sq_security_blocker', 0), result.get('sq_security_high', 0),
                    result.get('sq_security_medium', 0),
                    result.get('sq_security_low', 0), result.get('sq_security_info', 0),
                    result.get('sq_reliability_blocker', 0), result.get('sq_reliability_high', 0),
                    result.get('sq_reliability_medium', 0),
                    result.get('sq_reliability_low', 0), result.get('sq_reliability_info', 0),
                    result.get('sq_maintainability_blocker', 0), result.get('sq_maintainability_high', 0),
                    result.get('sq_maintainability_medium', 0),
                    result.get('sq_maintainability_low', 0), result.get('sq_maintainability_info', 0),
                    all_messages_str
                ])

        difficulty_str = f"({difficulty_filter})" if difficulty_filter else ""
        print(f"  Detailed results{difficulty_str} saved to: {output_file}")

    def get_all_llms_from_json(self, submission_dir: str = "submissions") -> List[str]:
        """Get all LLM names from submissions folder"""
        llm_folders = [
            d for d in os.listdir(submission_dir)
            if os.path.isdir(os.path.join(submission_dir, d)) and not d.startswith('.')
        ]
        return sorted(llm_folders)

    def save_summary_to_csv(self, output_file: str, append: bool = False):
        """Save summary results to CSV file"""
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        file_exists = os.path.exists(output_file)
        write_header = not file_exists or not append

        mode = 'a' if append and file_exists else 'w'

        with open(output_file, mode, newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            if write_header:
                writer.writerow([
                    'LLM', 'Total Codes', 'Correctness Rate (%)', 'Avg Issues Per Code', 'Total Issues',
                    'Severity: Blocker', 'Severity: Critical', 'Severity: Major', 'Severity: Minor', 'Severity: Info',
                    'Type: Bug', 'Type: Vulnerability', 'Type: Code Smell', 'Type: Security Hotspot',
                    'Attribute: Consistency', 'Attribute: Intentionality', 'Attribute: Adaptability',
                    'Attribute: Responsibility',
                    'SQ Security: Blocker', 'SQ Security: High', 'SQ Security: Medium', 'SQ Security: Low',
                    'SQ Security: Info',
                    'SQ Reliability: Blocker', 'SQ Reliability: High', 'SQ Reliability: Medium', 'SQ Reliability: Low',
                    'SQ Reliability: Info',
                    'SQ Maintainability: Blocker', 'SQ Maintainability: High', 'SQ Maintainability: Medium',
                    'SQ Maintainability: Low', 'SQ Maintainability: Info'
                ])

            for result in self.results:
                writer.writerow([
                    result['model'], result['total_codes'],
                    round(result.get('correctness_rate', 0), 2),
                    result['avg_issues_per_code'], result['total_issues'],
                    result.get('severity_blocker', 0), result.get('severity_critical', 0),
                    result.get('severity_major', 0),
                    result.get('severity_minor', 0), result.get('severity_info', 0),
                    result.get('type_bug', 0), result.get('type_vulnerability', 0), result.get('type_code_smell', 0),
                    result.get('type_security_hotspot', 0),
                    result.get('attr_consistency', 0), result.get('attr_intentionality', 0),
                    result.get('attr_adaptability', 0), result.get('attr_responsibility', 0),
                    result.get('sq_security_blocker', 0), result.get('sq_security_high', 0),
                    result.get('sq_security_medium', 0),
                    result.get('sq_security_low', 0), result.get('sq_security_info', 0),
                    result.get('sq_reliability_blocker', 0), result.get('sq_reliability_high', 0),
                    result.get('sq_reliability_medium', 0),
                    result.get('sq_reliability_low', 0), result.get('sq_reliability_info', 0),
                    result.get('sq_maintainability_blocker', 0), result.get('sq_maintainability_high', 0),
                    result.get('sq_maintainability_medium', 0),
                    result.get('sq_maintainability_low', 0), result.get('sq_maintainability_info', 0)
                ])

        if not append:
            print(f"\nSummary results saved to: {output_file}")
        else:
            print(f"\nSummary results appended to: {output_file}")

    def print_fetch_summary(self):
        """Print fetch summary"""
        print("\n" + "=" * 60)
        print("Data Fetch Summary")
        print("=" * 60)

        if self.fetch_errors:
            print(f"\n✖ Found {len(self.fetch_errors)} errors:")

            error_by_llm = defaultdict(list)
            for error in self.fetch_errors:
                if 'LLM:' in error:
                    llm = error.split(',')[0].replace('LLM: ', '').strip()
                    error_by_llm[llm].append(error)
                else:
                    error_by_llm['Other'].append(error)

            for llm, errors in sorted(error_by_llm.items()):
                print(f"\n  {llm} ({len(errors)} errors):")
                for err in errors:
                    print(f"    - {err}")
        else:
            print("\n No errors found")

        if self.fetch_errors:
            error_file = "sonarqube_result/fetch_errors.txt"
            os.makedirs("sonarqube_result", exist_ok=True)
            with open(error_file, 'w', encoding='utf-8') as f:
                f.write("SonarQube Data Fetch Error Report\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Total errors: {len(self.fetch_errors)}\n")
                f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("=" * 60 + "\n\n")
                for error in self.fetch_errors:
                    f.write(error + "\n")
            print(f"\nError report saved to: {error_file}")

    def run_analysis(self, organization_key: str, component_key: str, auth_token: str,
                     summary_output: str = "sonarqube_result/sonarqube_summary.csv"):
        """Analyze SonarQube results (single project containing multiple LLMs)"""
        print("=" * 60)
        print("Starting SonarQube Result Analysis")
        print("=" * 60)

        self.load_code_stats_from_json()

        all_llms_from_json = self.get_all_llms_from_json()
        print(f"\nFound {len(all_llms_from_json)} LLMs from JSON files")

        print(f"\nVerifying if LLM components exist in SonarQube...")
        valid_llms = []
        invalid_llms = []

        for llm_name in all_llms_from_json:
            if self.verify_llm_component_exists(organization_key, component_key, auth_token, llm_name):
                valid_llms.append(llm_name)
            else:
                invalid_llms.append(llm_name)

        if invalid_llms:
            print(f"\n  The following {len(invalid_llms)} LLMs do not exist in SonarQube:")
            for llm in invalid_llms:
                print(f"  - {llm}")

        print(f"\n✓ Verification complete: {len(valid_llms)} valid LLMs, {len(invalid_llms)} invalid LLMs")

        issues = self.fetch_sonarqube_issues(organization_key, component_key, auth_token)

        llm_file_issues = self.parse_issues_by_llm_and_file(issues)

        print(f"\n{'=' * 60}")
        print(f"Found {len(llm_file_issues)} LLMs with issues from SonarQube")
        print(f"{'=' * 60}")

        llms_without_issues = set(all_llms_from_json) - set(llm_file_issues.keys())

        if llms_without_issues:
            print(f"\nFound {len(llms_without_issues)} LLMs without issues:")
            for llm in sorted(llms_without_issues):
                total_codes, correctness = self.get_code_stats(llm)
                in_sonarqube = "✓" if llm in valid_llms else "✗"
                print(f"  [{in_sonarqube}] {llm}: {total_codes} codes, {correctness:.2f}% correct")

        print(f"\n{'=' * 60}")
        print(f"Starting analysis for all {len(all_llms_from_json)} LLMs")
        print(f"{'=' * 60}")

        for idx, llm_name in enumerate(all_llms_from_json, 1):
            print(f"\n[{idx}/{len(all_llms_from_json)}] " + "=" * 50)

            if llm_name in llm_file_issues:
                results = self.analyze_llm_issues(llm_name, llm_file_issues[llm_name])
            else:
                print(f"\nAnalyzing LLM: {llm_name}")
                total_codes, correctness = self.get_code_stats(llm_name)
                print(f"  JSON statistics: {total_codes} codes, correctness rate: {correctness:.2f}%")
                print(f"  No issues found for this LLM")
                results = self.create_empty_results(llm_name)

            if results:
                self.results = []
                self.results.append(results['all'])

                for difficulty in ['easy', 'medium', 'hard', 'unknown']:
                    if difficulty in results:
                        self.results.append(results[difficulty])

                append_mode = (idx > 1)
                self.save_summary_to_csv(summary_output, append=append_mode)

        print("\n" + "=" * 60)
        print("SonarQube Result Analysis Completed!")
        print("=" * 60)

        print(f"\nFinal statistics:")
        print(f"  Total LLMs: {len(all_llms_from_json)}")
        print(f"  LLMs with issues: {len(llm_file_issues)}")
        print(f"  LLMs without issues: {len(llms_without_issues)}")
        print(f"  LLMs not in SonarQube: {len(invalid_llms)}")

        self.print_fetch_summary()


if __name__ == "__main__":
    checker = SonarQubeChecker()

    organization_key = "jueh-96"
    component_key = "JueH-96_submission_code_id"
    auth_token = "89a4595467b4861d31165623885336c84e216978"

    checker.run_analysis(
        organization_key=organization_key,
        component_key=component_key,
        auth_token=auth_token,
        summary_output="sonarqube_result/sonarqube_summary.csv"
    )

    print("\nCompleted!")