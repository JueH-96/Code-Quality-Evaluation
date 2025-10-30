def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))

    counts = {}

    for i in range(m):
        current_votes = a[:i+1]
        current_counts = {}
        for vote in current_votes:
            current_counts[vote] = current_counts.get(vote, 0) + 1

        max_votes = -1
        winner = -1

        for candidate in range(1, n + 1):
            count = current_counts.get(candidate, 0)
            if count > max_votes:
                max_votes = count
                winner = candidate
            elif count == max_votes and winner != -1 and candidate < winner:
                winner = candidate
            elif count > 0 and winner == -1:
                max_votes = count
                winner = candidate
            elif count == max_votes and winner == -1:
                winner = candidate

        print(winner)

solve()