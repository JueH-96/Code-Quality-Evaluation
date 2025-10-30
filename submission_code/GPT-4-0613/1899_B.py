import sys
from typing import List

def max_diff(t: int, test_cases: List[List[int]]) -> List[int]:
    res = []
    for i in range(t):
        n, a = test_cases[i]
        a.sort()
        mid = sum(a) - a[0]
        for j in range(1, n//2 + 1):
            mid -= a[j-1]
            mid += a[n-j]
            res.append(abs(2*mid - sum(a)))
    return res

def from_input_string(input_string: str) -> Tuple[int, List[List[int]]]:
    lines = input_string.strip().split("
")
    t = int(lines[0])
    tests = []
    for x in range(1, len(lines[1:]), 2):
        n = int(lines[x])
        a = list(map(int, lines[x+1].strip().split()))
        tests.append((n, a))
    return t, tests

def to_input_string(inputs: Tuple[int, List[List[int]]]) -> str:
    t, tests = inputs
    res = []
    res.append(str(t))
    for n, a in tests:
        res.append(str(n))
        res.append(" ".join(str(x) for x in a))
    return "
".join(res)

def from_output_string(output_string: str) -> List[int]:
    lines = output_string.strip().split("
")
    return [int(x) for x in lines]

def to_output_string(output: List[int]) -> str:
    res = [str(x) for x in output]
    return "
".join(res)

EXAMPLES_PROVIDED = [{'input': '5
2
1 2
6
10 2 3 6 1 3
4
1000000000 1000000000 1000000000 1000000000
15
60978 82265 78961 56708 39846 31071 4913 4769 29092 91348 64119 72421 98405 222 14294
8
19957 69913 37531 96991 57838 21008 14207 19198
', 'output': '1
9
0
189114
112141
'}, {'input': '1
2
1 2
', 'output': '1
'}, {'input': '1
4
1000000000 1000000000 1000000000 1000000000
', 'output': '0
'}, {'input': '1
6
10 2 3 6 1 3
', 'output': '9
'}, {'input': '1
8
19957 69913 37531 96991 57838 21008 14207 19198
', 'output': '112141
'}, {'input': '1
15
60978 82265 78961 56708 39846 31071 4913 4769 29092 91348 64119 72421 98405 222 14294
', 'output': '189114
'}]

for example in EXAMPLES_PROVIDED:
    input_str = example['input']
    expected_output_str = example['output']
    inputs = from_input_string(input_str)
    expected_output = from_output_string(expected_output_str)

    # Ensure the function is correct
    assert max_diff(*inputs) == expected_output

    # Ensure str -> variable -> str consistency, while ignoring leading/trailing white space
    assert input_str.strip() == to_input_string(from_input_string(input_str)).strip()
    assert expected_output_str.strip() == to_output_string(from_output_string(expected_output_str)).strip()

    # Ensure variable -> str -> variable consistency
    assert inputs == from_input_string(to_input_string(inputs))
    assert expected_output == from_output_string(to_output_string(expected_output))