# YOUR CODE HERE
import sys

def is_subsequence(s, t):
    it = iter(t)
    return all(c in it for c in s)

def count_subsequence_pairs(N, T, S):
    count = 0
    prefix_count = [0] * (len(T) + 1)
    suffix_count = [0] * (len(T) + 1)
    
    for s in S:
        j = 0
        for c in s:
            if j < len(T) and c == T[j]:
                j += 1
        prefix_count[j] += 1
    
    for s in S:
        j = len(T) - 1
        for c in reversed(s):
            if j >= 0 and c == T[j]:
                j -= 1
        suffix_count[j] += 1
    
    for i in range(len(T) + 1):
        count += prefix_count[i] * suffix_count[i]
    
    return count

input = sys.stdin.read
data = input().split()
N = int(data[0])
T = data[1]
S = data[2:]

result = count_subsequence_pairs(N, T, S)
print(result)