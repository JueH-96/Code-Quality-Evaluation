import sys
from collections import deque

def main():
    N = int(sys.stdin.readline())
    A = []
    B = []
    sum_B = 0
    for _ in range(N):
        a, b = map(int, sys.stdin.readline().split())
        A.append(a)
        B.append(b)
        sum_B += b
    if sum_B % 3 != 0:
        print(-1)
        return
    T = sum_B // 3

    # Compute initial deltas
    d1 = 0
    d2 = 0
    d3 = 0
    for i in range(N):
        a = A[i]
        if a == 1:
            d1 += B[i]
        elif a == 2:
            d2 += B[i]
        else:
            d3 += B[i]

    # BFS setup
    visited = set()
    queue = deque()
    initial_state = (d1, d2)
    queue.append((d1, d2, 0))
    visited.add((d1, d2))

    while queue:
        current_d1, current_d2, moves = queue.popleft()

        if current_d1 == 0 and current_d2 == 0:
            print(moves)
            return

        # Generate all possible next states by moving each person
        for i in range(N):
            a = A[i]
            b = B[i]
            new_d1, new_d2, new_d3 = current_d1, current_d2, current_d3

            if a == 1:
                new_d1 -= b
                new_d2 += b
                new_d3 = T - (new_d1 + new_d2)
            elif a == 2:
                new_d2 -= b
                new_d1 += b
                new_d3 = T - (new_d1 + new_d2)
            else:
                new_d3 -= b
                new_d1 += b
                new_d2 = T - (new_d1 + new_d3)

            new_state = (new_d1, new_d2)
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_d1, new_d2, moves + 1))

    # If no solution found
    print(-1)

if __name__ == "__main__":
    main()