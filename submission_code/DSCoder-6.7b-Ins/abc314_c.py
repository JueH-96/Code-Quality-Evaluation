def solve():
    N, M = map(int, input().split())
    S = input()
    C = list(map(int, input().split()))

    color_positions = [[] for _ in range(M+1)]
    for i in range(N):
        color_positions[C[i]].append(i)

    for color in range(1, M+1):
        positions = color_positions[color]
        if positions:
            last_position = positions[-1]
            S = S[:last_position] + S[last_position+1:] + S[last_position]
            color_positions[color].pop()
            if color_positions[color]:
                color_positions[color].append(color_positions[color][0] - 1)

    print(S)

solve()