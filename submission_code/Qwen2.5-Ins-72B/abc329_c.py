# YOUR CODE HERE
import sys

def count_repeated_substrings(N, S):
    count = 0
    i = 0
    while i < N:
        j = i
        while j < N and S[j] == S[i]:
            j += 1
        length = j - i
        count += (length * (length + 1)) // 2
        i = j
    return count

N = int(input().strip())
S = input().strip()

result = count_repeated_substrings(N, S)
print(result)