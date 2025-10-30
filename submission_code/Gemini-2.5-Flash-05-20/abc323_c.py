# YOUR CODE HERE
import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    S = [sys.stdin.readline().strip() for _ in range(N)]

    # Calculate initial scores for all players
    # player_current_scores[i] stores the score for player (i+1)
    player_current_scores = [0] * N
    for i in range(N):
        current_score = i + 1  # Bonus score for player (i+1)
        for j in range(M):
            if S[i][j] == 'o':
                current_score += A[j]
        player_current_scores[i] = current_score

    # Process each player
    results = []
    for k in range(N): # k is the 0-indexed current player
        # Find the maximum score among all *other* players
        max_other_score = 0
        for i in range(N):
            if i != k:
                max_other_score = max(max_other_score, player_current_scores[i])

        # The target score player k needs to exceed
        target_score_for_k = max_other_score + 1

        # If player k's current score is already sufficient
        if player_current_scores[k] >= target_score_for_k:
            results.append(0)
            continue

        # Get scores of problems player k has not solved yet
        unsolved_problems_scores = []
        for j in range(M):
            if S[k][j] == 'x': # Problem j is unsolved for player k
                unsolved_problems_scores.append(A[j])

        # Sort unsolved problems by score in descending order
        unsolved_problems_scores.sort(reverse=True)

        # Simulate solving problems greedily
        current_score_with_additions = player_current_scores[k]
        problems_to_solve_count = 0

        for problem_score in unsolved_problems_scores:
            current_score_with_additions += problem_score
            problems_to_solve_count += 1
            if current_score_with_additions >= target_score_for_k:
                break # Player k has exceeded all other players' scores

        results.append(problems_to_solve_count)

    # Print all results
    for res in results:
        sys.stdout.write(str(res) + '
')

# Call the solve function to run the program
solve()