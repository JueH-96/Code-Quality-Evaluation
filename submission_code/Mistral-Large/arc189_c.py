import sys
input = sys.stdin.read

def solve():
    data = input().split()
    index = 0

    N = int(data[index])
    X = int(data[index + 1])
    index += 2

    A = [int(data[index + i]) for i in range(N)]
    index += N

    B = [int(data[index + i]) for i in range(N)]
    index += N

    P = [int(data[index + i]) for i in range(N)]
    index += N

    Q = [int(data[index + i]) for i in range(N)]

    # Convert P and Q to 0-based index
    P = [x - 1 for x in P]
    Q = [x - 1 for x in Q]
    X -= 1

    # Check if we can move all balls to box X
    visited = [False] * N
    stack = [X]
    while stack:
        node = stack.pop()
        if not visited[node]:
            visited[node] = True
            stack.append(P[node])
            stack.append(Q[node])

    if any(visited[i] == False and (A[i] > 0 or B[i] > 0) for i in range(N)):
        print(-1)
        return

    # Count the minimum number of operations
    operations = 0
    for i in range(N):
        if i != X and (A[i] > 0 or B[i] > 0):
            operations += 1

    print(operations)

if __name__ == "__main__":
    solve()