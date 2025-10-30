# YOUR CODE HERE
import sys

def solve():
    """
    Reads input, calculates the minimum problems to solve for each player, and prints the results.
    """
    # Read N and M, the number of players and problems
    try:
        line = sys.stdin.readline()
        if not line: return
        N, M = map(int, line.split())
    except (IOError, ValueError):
        return

    # Read the list of scores for each problem
    A = list(map(int, sys.stdin.readline().split()))

    # Read the solved status string for each player
    S_list = [sys.stdin.readline().strip() for _ in range(N)]

    # --- PART 1: Calculate initial scores ---
    # The score is the sum of solved problem scores plus a bonus equal to the player's 1-based index.
    scores = []
    for i in range(N):
        # Player i (0-indexed) corresponds to player "i+1" (1-indexed). Bonus is i+1.
        current_score = i + 1
        for j in range(M):
            if S_list[i][j] == 'o':
                current_score += A[j]
        scores.append(current_score)

    # --- PART 2: Determine problems to solve for each player ---
    # For each player, calculate the minimum number of additional problems they must solve
    # to have a score strictly greater than every other player's current score.
    for i in range(N):
        my_score = scores[i]
        
        # Find the maximum score among all other players. This is the score to beat.
        max_other_score = 0
        for j in range(N):
            if i != j:
                max_other_score = max(max_other_score, scores[j])

        # If the player's score is already the highest, they need to solve 0 more problems.
        if my_score > max_other_score:
            print(0)
            continue

        # Identify the problems the current player has not yet solved.
        unsolved_problems_scores = []
        for j in range(M):
            if S_list[i][j] == 'x':
                unsolved_problems_scores.append(A[j])
        
        # To solve the minimum number of problems, we use a greedy approach:
        # solve the problems with the highest scores first.
        unsolved_problems_scores.sort(reverse=True)
        
        # Calculate the total score needed to surpass the top opponent.
        score_to_gain = (max_other_score - my_score) + 1
        
        # Greedily "solve" problems and count how many it takes.
        problems_to_solve_count = 0
        gained_score_total = 0
        for problem_score in unsolved_problems_scores:
            gained_score_total += problem_score
            problems_to_solve_count += 1
            # Once we have gained enough score, we can stop.
            if gained_score_total >= score_to_gain:
                break
                
        print(problems_to_solve_count)

solve()