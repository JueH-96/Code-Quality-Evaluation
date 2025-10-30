def solve():
    n, t = map(int, input().split())
    updates = []
    for _ in range(t):
        updates.append(list(map(int, input().split())))

    scores = [0] * n

    for i in range(t):
        player_index, score_increase = updates[i]
        scores[player_index - 1] += score_increase
        unique_scores = set(scores)
        print(len(unique_scores))

solve()