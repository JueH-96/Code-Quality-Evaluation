def main():
    import sys

    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    A = list(map(int, input_data[2:2+M]))
    S = input_data[2+M:]

    # Precompute the current score for each player
    # current_score[i] = sum_of_solved_problem_scores + (i+1) as the bonus
    current_score = [0]*N
    solved_problems = []
    idx = 0
    for i in range(N):
        s = S[i]
        score_sum = 0
        for j in range(M):
            if s[j] == 'o':
                score_sum += A[j]
        current_score[i] = score_sum + (i+1)
        solved_problems.append(s)

    # For each player i, find the minimum number of new problems (among the ones not solved)
    # that must be solved to exceed all other players' current scores
    for i in range(N):
        # Highest score among the other players
        max_other = max(current_score[:i] + current_score[i+1:])

        # If this player's current score already exceeds the max of others, answer = 0
        if current_score[i] > max_other:
            print(0)
            continue

        need = max_other + 1 - current_score[i]

        # Gather the scores for not-solved problems for player i
        not_solved_scores = []
        for j in range(M):
            if solved_problems[i][j] == 'x':
                not_solved_scores.append(A[j])
        # Sort them in descending order to minimize the number of problems needed
        not_solved_scores.sort(reverse=True)

        count = 0
        gained = 0
        for score in not_solved_scores:
            gained += score
            count += 1
            if gained >= need:
                print(count)
                break

# Do not forget to call main
if __name__ == "__main__":
    main()