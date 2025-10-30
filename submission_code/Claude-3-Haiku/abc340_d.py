import sys

N = int(sys.stdin.readline().strip())
stages = []
for _ in range(N-1):
    A, B, X = map(int, sys.stdin.readline().strip().split())
    stages.append((A, B, X))

def solve():
    time = 0
    current_stage = 1
    while current_stage < N:
        min_time = float('inf')
        next_stage = None
        for i, (A, B, X) in enumerate(stages):
            if i+1 == current_stage:
                min_time = min(min_time, A)
                next_stage = i+2
            elif X == current_stage:
                min_time = min(min_time, B)
                next_stage = X
        time += min_time
        current_stage = next_stage
    return time

print(solve())