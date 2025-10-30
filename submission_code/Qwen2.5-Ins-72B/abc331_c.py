# YOUR CODE HERE
import sys

def solve():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    total_sum = sum(A)
    prefix_sum = 0
    result = []
    
    for i in range(N):
        prefix_sum += A[i]
        result.append(total_sum - prefix_sum + A[i])
    
    for i in range(N):
        result[i] = total_sum - result[i] - A[i]
    
    print(" ".join(map(str, result)))

solve()