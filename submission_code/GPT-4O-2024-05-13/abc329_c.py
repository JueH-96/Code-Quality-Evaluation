# YOUR CODE HERE
def count_repeated_substrings(N, S):
    count = 0
    i = 0
    while i < N:
        j = i
        while j < N and S[j] == S[i]:
            j += 1
        length = j - i
        count += length * (length + 1) // 2
        i = j
    return count

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
S = data[1]
print(count_repeated_substrings(N, S))