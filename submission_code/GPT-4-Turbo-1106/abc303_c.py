def can_complete_moves(N, M, H, K, S, items):
    x, y = 0, 0
    health = H
    items_dict = {tuple(item): True for item in items}

    for move in S:
        if move == 'R':
            x += 1
        elif move == 'L':
            x -= 1
        elif move == 'U':
            y += 1
        elif move == 'D':
            y -= 1

        health -= 1

        if health < 0:
            return "No"

        if (x, y) in items_dict and health < K:
            health = K

    return "Yes"

# Read input
N, M, H, K = map(int, input().split())
S = input().strip()
items = [tuple(map(int, input().split())) for _ in range(M)]

# Solve the problem and print the result
result = can_complete_moves(N, M, H, K, S, items)
print(result)