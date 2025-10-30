import sys
import math
from collections import Counter

def min_operations_to_divide_by_k(t, test_cases):
    results = []
    for n, k, a in test_cases:
        if k == 2:
            # Count the number of odd numbers
            odd_count = sum(1 for x in a if x % 2 != 0)
            results.append(odd_count)
        elif k == 3:
            # Count the number of numbers not divisible by 3
            not_div_by_3_count = sum(1 for x in a if x % 3 != 0)
            results.append(not_div_by_3_count)
        elif k == 4:
            # Count the number of numbers not divisible by 2 and 4
            not_div_by_2_count = sum(1 for x in a if x % 2 != 0)
            not_div_by_4_count = sum(1 for x in a if x % 4 != 0)
            results.append(min(not_div_by_2_count + not_div_by_4_count, not_div_by_4_count + 1))
        elif k == 5:
            # Count the number of numbers not divisible by 5
            not_div_by_5_count = sum(1 for x in a if x % 5 != 0)
            results.append(not_div_by_5_count)
    return results

def main():
    input = sys.stdin.read
    data = input().split()

    index = 0
    t = int(data[index])
    index += 1
    test_cases = []

    for _ in range(t):
        n = int(data[index])
        k = int(data[index + 1])
        index += 2
        a = list(map(int, data[index:index + n]))
        index += n
        test_cases.append((n, k, a))

    results = min_operations_to_divide_by_k(t, test_cases)
    for result in results:
        sys.stdout.write(f"{result}
")

if __name__ == "__main__":
    main()