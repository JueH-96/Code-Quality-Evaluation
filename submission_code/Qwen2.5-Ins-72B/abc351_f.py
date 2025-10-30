# YOUR CODE HERE
import sys

def solve():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    prefix_sum = [0] * N
    suffix_sum = [0] * N
    
    for i in range(1, N):
        prefix_sum[i] = prefix_sum[i-1] + A[i-1]
    
    for i in range(N-2, -1, -1):
        suffix_sum[i] = suffix_sum[i+1] + A[i+1]
    
    result = 0
    for i in range(N):
        result += A[i] * (N - 1 - i) - suffix_sum[i]
        result += prefix_sum[i] - A[i] * i
    
    print(result // 2)

if __name__ == "__main__":
    solve()