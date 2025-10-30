import sys
from itertools import combinations

def count_arithmetic_subsequences(sequence, length):
    """Count the number of arithmetic subsequences of a given length."""
    count = 0
    for subsequence in combinations(sequence, length):
        if is_arithmetic(subsequence):
            count += 1
    return count

def is_arithmetic(subsequence):
    """Check if a subsequence is arithmetic."""
    diff = subsequence[1] - subsequence[0]
    for i in range(2, len(subsequence)):
        if subsequence[i] - subsequence[i-1] != diff:
            return False
    return True

def main():
    """Read input and print the number of arithmetic subsequences for each length."""
    n = int(sys.stdin.readline())
    sequence = list(map(int, sys.stdin.readline().split()))
    result = []
    for k in range(1, n + 1):
        count = count_arithmetic_subsequences(sequence, k)
        result.append(count)
    print(*result)

if __name__ == "__main__":
    main()