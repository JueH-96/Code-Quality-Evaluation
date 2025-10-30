import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    
    left_distinct = [0] * N
    seen = set()
    for i in range(N):
        seen.add(A[i])
        left_distinct[i] = len(seen)
    
    right_distinct = [0] * N
    seen = set()
    for i in range(N-1, -1, -1):
        seen.add(A[i])
        right_distinct[i] = len(seen)
    
    max_sum = 0
    for i in range(N-1):
        current_sum = left_distinct[i] + right_distinct[i+1]
        if current_sum > max_sum:
            max_sum = current_sum
    
    print(max_sum)

solve()