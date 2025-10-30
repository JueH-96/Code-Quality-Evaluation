# YOUR CODE HERE
import sys

def count_distinct_substrings(s):
    substrings = set()
    n = len(s)
    for i in range(n):
        for j in range(i + 1, n + 1):
            substrings.add(s[i:j])
    return len(substrings)

input = sys.stdin.read().strip()
print(count_distinct_substrings(input))