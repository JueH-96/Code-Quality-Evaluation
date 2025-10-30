def solve():
    n, t = map(int, input().split())
    updates = []
    for _ in range(t):
        updates.append(list(map(int, input().split())))
    
    scores = [0] * n
    
    for i in range(t):
        player_index, score_increment = updates[i]
        scores[player_index - 1] += score_increment
        distinct_scores_count = len(set(scores))
        print(distinct_scores_count)

if __name__ == '__main__':
    solve()