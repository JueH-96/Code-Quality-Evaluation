def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    A = list(map(int, data[2:2+M]))
    S = data[2+M:2+M+N]

    # Compute current totals for each player:
    # total_score[i] = (sum of scores of already solved problems) + (bonus of i+1)
    total_score = []
    for i in range(N):
        solved_sum = 0
        for j in range(M):
            if S[i][j] == 'o':
                solved_sum += A[j]
        total_score.append(solved_sum + (i+1))

    # For each player i, compute the minimum number of unsolved problems to exceed
    # every other player's current total score.
    for i in range(N):
        # The score we need to exceed
        max_other = max(total_score[j] for j in range(N) if j != i)
        # If player i is already ahead, no additional problems needed
        if total_score[i] > max_other:
            print(0)
            continue

        # Gather the scores of unsolved problems for player i and sort in descending order
        unsolved_scores = []
        for j in range(M):
            if S[i][j] == 'x':
                unsolved_scores.append(A[j])
        unsolved_scores.sort(reverse=True)

        # Add problems one-by-one (largest first) until the player's total exceeds max_other
        needed = max_other - total_score[i] + 1
        current_sum = 0
        count = 0
        for score in unsolved_scores:
            current_sum += score
            count += 1
            if current_sum >= needed:
                print(count)
                break

# Do not forget to call main() at the end
if __name__ == "__main__":
    main()