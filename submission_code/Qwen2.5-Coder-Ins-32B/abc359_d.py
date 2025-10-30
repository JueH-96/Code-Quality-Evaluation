# YOUR CODE HERE
MOD = 998244353

def is_palindrome(s):
    return s == s[::-1]

def count_good_strings(N, K, S):
    def backtrack(index, current):
        if index == N:
            for i in range(N - K + 1):
                if is_palindrome(current[i:i + K]):
                    return 0
            return 1
        if S[index] != '?':
            return backtrack(index + 1, current + S[index])
        else:
            return (backtrack(index + 1, current + 'A') + backtrack(index + 1, current + 'B')) % MOD

    return backtrack(0, '')

import sys
input = sys.stdin.read().split()
N = int(input[0])
K = int(input[1])
S = input[2]

print(count_good_strings(N, K, S))