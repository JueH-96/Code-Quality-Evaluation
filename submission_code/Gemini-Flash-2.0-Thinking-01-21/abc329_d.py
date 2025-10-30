def solve():
    n, m = map(int, input().split())
    votes = list(map(int, input().split()))
    vote_counts = [0] * n
    for i in range(m):
        vote = votes[i]
        vote_counts[vote - 1] += 1
        max_votes = -1
        winner = -1
        for candidate_number in range(1, n + 1):
            current_votes = vote_counts[candidate_number - 1]
            if current_votes > max_votes:
                max_votes = current_votes
                winner = candidate_number
        print(winner)

if __name__ == '__main__':
    solve()