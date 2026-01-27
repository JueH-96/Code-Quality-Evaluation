import re
import json
import os
import sys


def parse_comprehensive_table(latex_file):
    """Parse the comprehensive table with new format (48 columns: 16 metrics × 3 difficulties)"""
    if not os.path.exists(latex_file):
        raise FileNotFoundError(f"Cannot find file: {latex_file}")

    with open(latex_file, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')

    llms = []
    easy_data = []
    medium_data = []
    hard_data = []

    # Parse data rows
    in_data = False
    for line in lines:
        line = line.strip()

        # Start parsing after the second \midrule
        if '\\midrule' in line:
            in_data = True
            continue

        # Stop at \bottomrule
        if '\\bottomrule' in line or '\\end{tabular}' in line:
            break

        # Skip header lines and empty lines
        if not line or '\\textbf{LLM}' in line or (line.startswith('LLM') and '&' in line and 'E &' in line):
            continue

        # Parse data row
        if '&' in line and in_data:
            # Split by & and clean up
            parts = [p.strip() for p in line.split('&')]

            # Remove trailing \\ from last element
            if parts:
                parts[-1] = parts[-1].replace('\\\\', '').strip()

            # Should have LLM name + 48 values (16 metrics × 3 difficulties)
            if len(parts) >= 49:
                llm_name = parts[0].strip()
                if not llm_name or llm_name.startswith('\\') or llm_name == 'LLM':
                    continue

                llms.append(llm_name)

                try:
                    # Extract values for each difficulty
                    # Format: Conv(E,M,H), Ref(E,M,H), Warn(E,M,H), Err(E,M,H),
                    #         B-Issue(E,M,H), R-Comp(E,M,H), R-MI(E,M,H), R-Func(E,M,H),
                    #         R-LOC(E,M,H), R-LLOC(E,M,H), R-SLOC(E,M,H), R-CMT(E,M,H),
                    #         R-BLN(E,M,H), SQ-M(E,M,H), SQ-R(E,M,H), SQ-S(E,M,H)

                    values = [float(p) for p in parts[1:49]]

                    # Extract Easy, Medium, Hard separately
                    # Every 3 values are E, M, H for each metric
                    easy_row = []
                    medium_row = []
                    hard_row = []

                    for i in range(0, 48, 3):
                        easy_row.append(values[i])  # E
                        medium_row.append(values[i + 1])  # M
                        hard_row.append(values[i + 2])  # H

                    easy_data.append(easy_row)
                    medium_data.append(medium_row)
                    hard_data.append(hard_row)

                except (ValueError, IndexError) as e:
                    print(f"Warning: Error parsing row for {llm_name}: {e}")
                    print(f"  Parts length: {len(parts)}")
                    continue

    return {
        'llms': llms,
        'easy': easy_data,
        'medium': medium_data,
        'hard': hard_data
    }


def parse_overall_table(latex_file):
    """Parse the overall performance table (16 columns)"""
    if not os.path.exists(latex_file):
        raise FileNotFoundError(f"Cannot find file: {latex_file}")

    with open(latex_file, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')

    llms = []
    data = []

    # Find the start of data (after the header with "Conv")
    data_start = False
    for i, line in enumerate(lines):
        line = line.strip()

        # Look for the header row with metric names
        if '\\textbf{Conv}' in line and '\\textbf{Ref}' in line:
            data_start = True
            continue

        if not data_start:
            continue

        # Stop at the closing \hline
        if '\\hline' in line and data:
            break

        # Skip \hline lines and empty lines
        if '\\hline' in line or not line:
            continue

        # Parse data rows
        if '&' in line:
            parts = [p.strip() for p in line.split('&')]

            # Remove trailing \\ from last element
            if parts:
                parts[-1] = parts[-1].replace('\\\\', '').strip()

            # Should have LLM name + 16 values (updated from 10)
            if len(parts) >= 17:
                llm_name = parts[0].strip()
                if not llm_name or '\\textbf' in llm_name:
                    continue

                llms.append(llm_name)

                try:
                    row_data = [float(parts[i]) for i in range(1, 17)]
                    data.append(row_data)
                except (ValueError, IndexError) as e:
                    print(f"Warning: Error parsing row for {llm_name}: {e}")
                    print(f"  Expected 16 values, got {len(parts) - 1}")
                    continue

    return {
        'llms': llms,
        'data': data
    }


def find_tex_files():
    """Try to find the tex files in current directory"""
    comprehensive_file = 'code_quality_comprehensive_table.tex'
    overall_file = 'code_quality_overall_table.tex'

    if os.path.exists(comprehensive_file) and os.path.exists(overall_file):
        return comprehensive_file, overall_file

    return None, None


if __name__ == '__main__':
    print("LaTeX Table Parser (Updated Format)")
    print("=" * 50)

    # Check command line arguments
    if len(sys.argv) == 3:
        comprehensive_file = sys.argv[1]
        overall_file = sys.argv[2]
        print(f"Using files from command line arguments:")
        print(f"  Comprehensive: {comprehensive_file}")
        print(f"  Overall: {overall_file}")
    else:
        # Try to find files automatically
        print("Searching for .tex files in current directory...")
        comprehensive_file, overall_file = find_tex_files()

    # Validate files exist
    if not comprehensive_file or not os.path.exists(comprehensive_file):
        print("\n❌ ERROR: Cannot find 'code_quality_comprehensive_table.tex'")
        print("\nPlease make sure the file is in one of these locations:")
        print("  1. Current directory")
        print("  2. Specify path: python parse_latex_tables.py <comprehensive.tex> <overall.tex>")
        print(f"\nCurrent directory: {os.getcwd()}")
        print("Files in current directory:")
        for f in os.listdir('.'):
            if f.endswith('.tex'):
                print(f"  - {f}")
        sys.exit(1)

    if not overall_file or not os.path.exists(overall_file):
        print("\n❌ ERROR: Cannot find 'code_quality_overall_table.tex'")
        print("\nPlease make sure the file is in one of these locations:")
        print("  1. Current directory")
        print("  2. Specify path: python parse_latex_tables.py <comprehensive.tex> <overall.tex>")
        print(f"\nCurrent directory: {os.getcwd()}")
        print("Files in current directory:")
        for f in os.listdir('.'):
            if f.endswith('.tex'):
                print(f"  - {f}")
        sys.exit(1)

    print(f"\n✓ Found comprehensive table: {comprehensive_file}")
    print(f"✓ Found overall table: {overall_file}")
    print("\nParsing tables...")

    try:
        # Parse both tables
        comprehensive = parse_comprehensive_table(comprehensive_file)
        print(f"✓ Parsed comprehensive table: {len(comprehensive['llms'])} LLMs")
        print(f"  - Each difficulty level has {len(comprehensive['easy'][0]) if comprehensive['easy'] else 0} metrics")

        overall = parse_overall_table(overall_file)
        print(f"✓ Parsed overall table: {len(overall['llms'])} LLMs")
        print(f"  - Each LLM has {len(overall['data'][0]) if overall['data'] else 0} metrics")

        # Combine into one JSON file
        output = {
            'comprehensive': comprehensive,
            'overall': overall
        }

        output_file = 'benchmark_data.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)

        print(f"\n✅ SUCCESS! Data saved to: {output_file}")
        print(f"   Location: {os.path.abspath(output_file)}")
        print(f"\n📊 Summary:")
        print(f"   Comprehensive: {len(comprehensive['llms'])} LLMs × 16 metrics × 3 difficulties")
        print(f"   Overall: {len(overall['llms'])} LLMs × 16 metrics")

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)