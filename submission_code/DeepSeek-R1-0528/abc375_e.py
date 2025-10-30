def main():
    n = int(input().strip())
    groups = [[], [], []]
    total = 0
    for _ in range(n):
        a, b = map(int, input().split())
        groups[a-1].append(b)
        total += b

    if total % 3 != 0:
        print(-1)
        return

    target = total // 3
    F = [[-10**9] * (target + 1) for _ in range(target + 1)]
    F[0][0] = 0

    for g in range(3):
        for b in groups[g]:
            new_F = [[-10**9] * (target + 1) for _ in range(target + 1)]
            for a1 in range(target + 1):
                for a2 in range(target + 1):
                    if F[a1][a2] == -10**9:
                        continue
                    # Assign to team1
                    if a1 + b <= target:
                        non_moves = F[a1][a2] + (1 if g == 0 else 0)
                        if non_moves > new_F[a1 + b][a2]:
                            new_F[a1 + b][a2] = non_moves
                    # Assign to team2
                    if a2 + b <= target:
                        non_moves = F[a1][a2] + (1 if g == 1 else 0)
                        if non_moves > new_F[a1][a2 + b]:
                            new_F[a1][a2 + b] = non_moves
                    # Assign to team3
                    non_moves = F[a1][a2] + (1 if g == 2 else 0)
                    if non_moves > new_F[a1][a2]:
                        new_F[a1][a2] = non_moves
            F = new_F

    if F[target][target] < 0:
        print(-1)
    else:
        print(n - F[target][target])

if __name__ == "__main__":
    main()