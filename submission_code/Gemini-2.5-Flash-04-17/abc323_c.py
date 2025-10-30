# YOUR CODE HERE
import sys

# Function to solve the problem
def solve():
    # Read N (number of players) and M (number of problems)
    N, M = map(int, sys.stdin.readline().split())

    # Read problem scores A (A_1, A_2, ..., A_M) -> stored as A[0], A[1], ..., A[M-1]
    # A[j] is the score of the j-th problem (0-indexed)
    A = list(map(int, sys.stdin.readline().split()))

    # Read player submission strings S (S_1, S_2, ..., S_N) -> stored as S[0], S[1], ..., S[N-1]
    # S[i] is the submission string for the i-th player (0-indexed)
    # S[i][j] == 'o' means player i solved problem j (0-indexed)
    S = [sys.stdin.readline().strip() for _ in range(N)]

    # 1. Calculate current scores for all players
    # current_scores[i] will store the current total score for player i (0-indexed)
    current_scores = []
    for i in range(N): # Iterate through players (0 to N-1)
        score = i + 1 # Add the bonus for player i+1 (1-indexed player number)
        for j in range(M): # Iterate through problems (0 to M-1)
            # If player i has solved problem j (represented by S[i][j] == 'o')
            # Note: S_i's j-th character corresponds to problem j (0-indexed in A)
            if S[i][j] == 'o':
                score += A[j] # Add the score of problem j
        current_scores.append(score)

    # 2. For each player, find the minimum number of additional problems they need to solve
    for i in range(N): # Iterate through players (0 to N-1)
        # Calculate the maximum current score among all other players
        # This generator expression creates scores of players k where k is not i.
        # max() finds the maximum value among these scores.
        # This handles the case N=2 correctly.
        max_other_scores = max(current_scores[k] for k in range(N) if k != i)

        # If the current player's score is already strictly greater than
        # the maximum score of any other player
        if current_scores[i] > max_other_scores:
            print(0) # They need 0 additional problems to be ahead
            continue # Move to the next player

        # If the current player is not already ahead, we need to find how many problems they
        # must solve to exceed the max_other_scores.

        # Find the problems player i has not yet solved and get their scores.
        unsolved_scores = []
        for j in range(M): # Iterate through problems (0 to M-1)
            # If player i has not solved problem j (represented by S[i][j] == 'x')
            if S[i][j] == 'x':
                unsolved_scores.append(A[j]) # Add the score of problem j to the list

        # To minimize the *number* of problems solved, the player should solve
        # the unsolved problems with the highest scores first.
        # Sort the scores of unsolved problems in descending order.
        unsolved_scores.sort(reverse=True)

        # Iteratively "solve" problems from the sorted list (highest score first)
        # and add their scores to the current score until the score exceeds max_other_scores.
        needed_count = 0 # Initialize the count of problems solved
        temp_score = current_scores[i] # Start with the current score of player i

        # Iterate through the sorted list of scores for unsolved problems
        # The loop will always find a solution because solving all problems is guaranteed to be sufficient.
        for score in unsolved_scores:
            temp_score += score # Add the score of the next problem solved
            needed_count += 1 # Increment the count of problems solved

            # Check if the temporary score (current score + added scores)
            # is now strictly greater than the maximum score of other players
            if temp_score > max_other_scores:
                print(needed_count) # Print the number of problems needed
                break # We found the minimum number of problems required, so stop and go to the next player

# Execute the solve function when the script is run
solve()