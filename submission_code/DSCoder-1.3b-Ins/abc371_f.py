import sys

def solve():
    N = int(sys.stdin.readline().strip())
    X = list(map(int, sys.stdin.readline().strip().split()))
    Q = int(sys.stdin.readline().strip())
    tasks = [list(map(int, sys.stdin.readline().strip().split())) for _ in range(Q)]

    # Calculate the minimum number of movements for each task
    movements = [0]*Q
    for i in range(Q):
        for j in range(N):
            if X[j] == tasks[i][1]:
                movements[i] = abs(j - tasks[i][0])
                break

    # Calculate the total number of movements
    total_movements = sum(movements)

    # Write the answer to stdout
    sys.stdout.write(str(total_movements) + '
')

solve()