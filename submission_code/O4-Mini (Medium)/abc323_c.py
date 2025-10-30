def main():
    import sys
    input = sys.stdin.readline

    N, M = map(int, input().split())
    A = list(map(int, input().split()))
    S = [input().strip() for _ in range(N)]

    # Precompute current total scores (including bonus i+1)
    curr_scores = []
    for i in range(N):
        solved_score = 0
        si = S[i]
        for j, ch in enumerate(si):
            if ch == 'o':
                solved_score += A[j]
        # bonus is (i+1)
        curr_scores.append(solved_score + (i + 1))

    # For each player i, compute answer
    for i in range(N):
        my_score = curr_scores[i]
        # maximum score among the other players
        max_other = 0
        for j in range(N):
            if j == i:
                continue
            if curr_scores[j] > max_other:
                max_other = curr_scores[j]

        # If already strictly greater, zero more solves needed
        if my_score > max_other:
            print(0)
            continue

        # Gather the scores of problems not yet solved by player i
        unsolved = []
        si = S[i]
        for j, ch in enumerate(si):
            if ch == 'x':
                unsolved.append(A[j])

        # Sort descending to pick largest first
        unsolved.sort(reverse=True)

        # Try taking k = 1..len(unsolved)
        added = 0
        ans = 0
        for k, score in enumerate(unsolved, start=1):
            added += score
            if my_score + added > max_other:
                ans = k
                break

        print(ans)

if __name__ == "__main__":
    main()