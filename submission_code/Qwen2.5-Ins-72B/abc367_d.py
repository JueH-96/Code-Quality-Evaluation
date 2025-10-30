# YOUR CODE HERE
import sys

def count_pairs(N, M, A):
    prefix_sum = [0] * N
    prefix_sum[0] = A[0]
    for i in range(1, N):
        prefix_sum[i] = (prefix_sum[i-1] + A[i]) % M
    
    count = [0] * M
    for i in range(N):
        count[prefix_sum[i]] += 1
    
    result = 0
    for i in range(M):
        result += count[i] * (count[i] - 1)
    
    for i in range(N):
        result += count[prefix_sum[i]] - 1
    
    return result // 2

input = sys.stdin.read
data = input().split()
N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:]))
print(count_pairs(N, M, A))