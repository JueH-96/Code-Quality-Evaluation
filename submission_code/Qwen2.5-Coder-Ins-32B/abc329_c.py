# YOUR CODE HERE
import sys

def count_repetitive_substrings(s):
    n = len(s)
    count = 0
    i = 0
    while i < n:
        j = i
        while j < n and s[j] == s[i]:
            j += 1
        length = j - i
        count += length * (length + 1) // 2
        i = j
    return count

input = sys.stdin.read().split()
N = int(input[0])
S = input[1]
print(count_repetitive_substrings(S))