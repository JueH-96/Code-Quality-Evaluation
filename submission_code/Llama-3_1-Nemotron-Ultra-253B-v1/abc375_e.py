import sys
from collections import defaultdict

def main():
    N = int(sys.stdin.readline())
    people = []
    S = [0, 0, 0]
    for _ in range(N):
        A, B = map(int, sys.stdin.readline().split())
        A -= 1  # 0-based index
        people.append((A, B))
        S[A] += B

    total = sum(S)
    if total % 3 != 0:
        print(-1)
        return
    T = total // 3
    tx = -(T - S[0])
    ty = -(T - S[1])

    # Generate movement options for each person
    movements = []
    for A, B in people:
        if A == 0:
            # Move to team 1 or 2
            opt1 = (B, -B)
            opt2 = (B, 0)
        elif A == 1:
            # Move to team 0 or 2
            opt1 = (-B, B)
            opt2 = (0, B)
        else:
            # Move to team 0 or 1
            opt1 = (-B, 0)
            opt2 = (0, -B)
        movements.append((opt1, opt2))

    # DP state: (x, y) -> minimal moves
    dp = defaultdict(lambda: float('inf'))
    dp[(0, 0)] = 0

    for (opt1, opt2) in movements:
        new_dp = defaultdict(lambda: float('inf'))
        for (x, y), cost in dp.items():
            # Option 1: do not move
            if new_dp[(x, y)] > cost:
                new_dp[(x, y)] = cost
            # Option 2: move opt1
            nx = x + opt1[0]
            ny = y + opt1[1]
            if new_dp[(nx, ny)] > cost + 1:
                new_dp[(nx, ny)] = cost + 1
            # Option 3: move opt2
            nx = x + opt2[0]
            ny = y + opt2[1]
            if new_dp[(nx, ny)] > cost + 1:
                new_dp[(nx, ny)] = cost + 1
        dp = new_dp

    target = (tx, ty)
    if dp[target] != float('inf'):
        print(dp[target])
    else:
        print(-1)

if __name__ == '__main__':
    main()