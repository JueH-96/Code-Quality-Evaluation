# YOUR CODE HERE
import sys
input = sys.stdin.read
def solve():
    N = int(input())
    X = list(map(int, input().split()))
    Q = int(input())
    tasks = [tuple(map(int, input().split())) for _ in range(Q)]
    
    prefix_sum = [0] * (N + 1)
    for i in range(N):
        prefix_sum[i + 1] = prefix_sum[i] + abs(X[i] - X[-1])
    
    def min_moves(task):
        person, goal = task
        person -= 1
        left = 0
        right = N - 1
        while left < right:
            mid = (left + right) // 2
            if X[mid] <= goal:
                left = mid + 1
            else:
                right = mid
        if X[left] == goal:
            return prefix_sum[left] + prefix_sum[N] - prefix_sum[left + 1]
        else:
            return min(prefix_sum[left] + abs(X[left] - goal), prefix_sum[left + 1] + abs(X[left + 1] - goal))
    
    total_moves = 0
    for task in tasks:
        total_moves += min_moves(task)
    
    print(total_moves)
solve()