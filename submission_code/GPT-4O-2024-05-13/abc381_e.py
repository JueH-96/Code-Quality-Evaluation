# YOUR CODE HERE
def is_11_22_string(subseq):
    n = len(subseq)
    if n % 2 == 0:
        return False
    mid = n // 2
    if subseq[mid] != '/':
        return False
    for i in range(mid):
        if subseq[i] != '1':
            return False
    for i in range(mid + 1, n):
        if subseq[i] != '2':
            return False
    return True

def max_11_22_subsequence_length(S, L, R):
    T = S[L-1:R]
    n = len(T)
    max_length = 0
    for length in range(1, n+1, 2):
        for i in range(n - length + 1):
            subseq = T[i:i+length]
            if is_11_22_string(subseq):
                max_length = max(max_length, length)
    return max_length

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
Q = int(data[1])
S = data[2]
queries = []
index = 3
for _ in range(Q):
    L = int(data[index])
    R = int(data[index + 1])
    queries.append((L, R))
    index += 2

results = []
for L, R in queries:
    results.append(max_11_22_subsequence_length(S, L, R))

for result in results:
    print(result)