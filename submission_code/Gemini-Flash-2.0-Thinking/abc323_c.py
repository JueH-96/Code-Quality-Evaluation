def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    s = [input() for _ in range(n)]

    def calculate_current_score(player_index):
        score = player_index + 1
        for j in range(m):
            if s[player_index][j] == 'o':
                score += a[j]
        return score

    current_scores = [calculate_current_score(i) for i in range(n)]

    for i in range(n):
        max_other_score = 0
        for j in range(n):
            if i != j:
                max_other_score = max(max_other_score, current_scores[j])

        if current_scores[i] > max_other_score:
            print(0)
            continue

        unsolved_problems_indices = [j for j in range(m) if s[i][j] == 'x']
        unsolved_problems_scores = sorted([a[j] for j in unsolved_problems_indices], reverse=True)
        num_unsolved = len(unsolved_problems_scores)

        for k in range(num_unsolved + 1):
            potential_increase = sum(unsolved_problems_scores[:k])
            potential_score = current_scores[i] + potential_increase
            if potential_score > max_other_score:
                print(k)
                break

solve()