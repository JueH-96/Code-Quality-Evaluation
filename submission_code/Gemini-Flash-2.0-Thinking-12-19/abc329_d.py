def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))

    counts = {}
    for i in range(m):
        vote = a[i]
        counts[vote] = counts.get(vote, 0) + 1

        max_votes = -1
        winner = -1
        for candidate in sorted(counts.keys()):
            if counts[candidate] > max_votes:
                max_votes = counts[candidate]
                winner = candidate
        print(winner)

solve()