# YOUR CODE HERE
import sys

def solve():
    # Read number of players N and number of problems M
    N, M = map(int, sys.stdin.readline().split())
    
    # Read problem scores A_1, ..., A_M
    # Store them in a 0-indexed list A
    A = list(map(int, sys.stdin.readline().split()))
    
    # Read the status string S_i for each player i
    S = []
    for _ in range(N):
        S.append(sys.stdin.readline().strip())

    # Calculate initial scores and identify unsolved problems for each player
    
    # scores[i] will store the current total score of player i+1 (using 0-based index i)
    scores = [0] * N 
    # unsolved_problems[i] will store a list of scores of problems not yet solved by player i+1
    unsolved_problems = [[] for _ in range(N)] 

    for i in range(N):
        # Player index i (0-based) corresponds to player number i+1 (1-based)
        # The bonus score for player i+1 is i+1
        current_score = i + 1 
        
        # This list will store the scores of problems player i+1 hasn't solved yet
        player_unsolved_scores = []
        
        for j in range(M):
            # Problem index j (0-based) corresponds to problem number j+1 (1-based)
            if S[i][j] == 'o':
                # If player i+1 solved problem j+1, add its score A[j] to their current score
                current_score += A[j]
            else:
                # If problem j+1 is unsolved by player i+1, add its score A[j] to the list
                # This list will be used later for the greedy strategy
                player_unsolved_scores.append(A[j]) 
        
        # Store the calculated total current score for player i+1
        scores[i] = current_score
        
        # Sort the scores of unsolved problems in descending order.
        # This is crucial for the greedy approach where we pick highest score problems first.
        player_unsolved_scores.sort(reverse=True)
        unsolved_problems[i] = player_unsolved_scores

    # List to store the final answer (minimum problems needed) for each player
    results = [] 
    
    # Iterate through each player to determine the minimum problems they need to solve
    for i in range(N):
        # Find the maximum score among all other players (j != i)
        max_other_score = 0
        for j in range(N):
            if i != j:
                max_other_score = max(max_other_score, scores[j])
        
        # Player i+1 needs their final score to be strictly greater than max_other_score.
        # The minimum integer score that satisfies this condition is max_other_score + 1.
        target_score = max_other_score + 1
        
        # Calculate the minimum additional score player i+1 needs to gain to reach target_score.
        score_needed = target_score - scores[i]

        # Check if player i+1's current score is already sufficient.
        # If score_needed <= 0, it means scores[i] >= target_score = max_other_score + 1,
        # which is equivalent to scores[i] > max_other_score.
        # In this case, player i+1 already exceeds all other players' scores, so 0 additional problems are needed.
        if score_needed <= 0:
            results.append(0)
            continue # Move to the next player

        # If additional score is needed (score_needed > 0):
        # Apply the greedy strategy: solve unsolved problems starting from the one with the highest score.
        
        problems_to_solve_count = 0 # Counter for the number of problems solved
        current_score_gain = 0 # Accumulator for the score gained from solving problems
        
        # Get the list of unsolved problem scores for player i+1, already sorted descending
        player_unsolved_scores_list = unsolved_problems[i] 

        # Iterate through the sorted scores of unsolved problems
        for problem_score in player_unsolved_scores_list:
             # Add the score of the current highest-scoring unsolved problem
             current_score_gain += problem_score 
             # Increment the count of problems solved
             problems_to_solve_count += 1 
             
             # Check if the total score gain is now sufficient to reach/exceed the needed score
             if current_score_gain >= score_needed:
                 # If yes, we have found the minimum number of problems required.
                 # Break the loop as we don't need to solve more problems.
                 break 
        
        # Append the minimum count found to the results list
        results.append(problems_to_solve_count)

    # Print the result for each player, one per line, as required by the problem statement.
    for res in results:
        print(res)

# Call the main function to execute the logic
solve()