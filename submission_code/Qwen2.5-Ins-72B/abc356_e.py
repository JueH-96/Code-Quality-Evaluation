# YOUR CODE HERE
import sys

def solve(N, A):
    result = 0
    for i in range(N - 1):
        for j in range(i + 1, N):
            max_val = max(A[i], A[j])
            min_val = min(A[i], A[j])
            result += max_val // min_val
    return result

input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:]))

print(solve(N, A))