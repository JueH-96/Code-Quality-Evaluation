# YOUR CODE HERE
def count_valid_pairs(N, T, strings):
    len_T = len(T)
    
    # Precompute prefix possibilities
    prefix_possible = [0] * (len_T + 1)
    for s in strings:
        m = len(s)
        dp = [0] * (len_T + 1)
        dp[0] = 1  # Empty prefix is always possible
        for char in s:
            for j in range(len_T - 1, -1, -1):
                if char == T[j]:
                    dp[j + 1] += dp[j]
        for j in range(len_T + 1):
            if dp[j] > 0:
                prefix_possible[j] += 1
    
    # Precompute suffix possibilities
    suffix_possible = [0] * (len_T + 1)
    for s in strings:
        m = len(s)
        dp = [0] * (len_T + 1)
        dp[len_T] = 1  # Full suffix is always possible
        for char in reversed(s):
            for j in range(len_T):
                if char == T[j]:
                    dp[j] += dp[j + 1]
        for j in range(len_T + 1):
            if dp[j] > 0:
                suffix_possible[j] += 1
    
    # Count valid pairs
    total_pairs = 0
    for k in range(len_T + 1):
        total_pairs += prefix_possible[k] * suffix_possible[k]
    
    return total_pairs

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
T = data[1]
strings = data[2:]

print(count_valid_pairs(N, T, strings))