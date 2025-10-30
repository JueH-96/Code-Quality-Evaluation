# YOUR CODE HERE
def is_11_22_string(substring):
    length = len(substring)
    if length % 2 == 0:
        return False
    mid = (length + 1) // 2
    if substring[mid - 1] != '/':
        return False
    if not all(c == '1' for c in substring[:mid - 1]):
        return False
    if not all(c == '2' for c in substring[mid:]):
        return False
    return True

def max_11_22_length(N, S):
    max_length = 0
    for i in range(N):
        for j in range(i + 1, N + 1):
            if is_11_22_string(S[i:j]):
                max_length = max(max_length, j - i)
    return max_length

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
S = data[1]

print(max_11_22_length(N, S))