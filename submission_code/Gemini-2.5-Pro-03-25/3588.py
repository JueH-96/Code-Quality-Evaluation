import sys 

# Set a higher recursion depth limit if using a recursive solution. Not necessary for this iterative solution.
# sys.setrecursionlimit(2000) 

MOD = 10**9 + 7
# Map creature characters to indices for easier processing
char_to_idx = {'F': 0, 'W': 1, 'E': 2}

class Solution:
  """
  This class provides a method to count the number of distinct sequences 
  Bob can use to beat Alice in a fantasy battle game, following specific rules 
  and constraints. It uses iterative dynamic programming to solve the problem efficiently.
  """
  def countWinningSequences(self, s: str) -> int:
    """
    Calculates the number of Bob's winning sequences modulo 10^9 + 7.

    A sequence is winning for Bob if his total score is strictly greater than Alice's total score
    after n rounds. Bob cannot summon the same creature in two consecutive rounds.

    Args:
        s: A string representing Alice's sequence of moves ('F', 'W', 'E'). The length of s is n.

    Returns:
        The number of Bob's distinct winning sequences modulo 10^9 + 7.
    """
    n = len(s)
    
    # Initialize DP table. dp[i][last_B_idx_shifted][shifted_score_diff] stores the number of ways
    # for Bob to complete rounds i through n-1 such that Bob eventually wins, given that
    # Bob's move in round i-1 was last_B_idx, and the score difference (Bob - Alice) up to
    # round i-1 is score_diff.
    # Dimensions:
    # (n+1) for round index i (0 to n)
    # 4 for last_B_idx_shifted (representing last_B_idx from -1 to 2; shifted by +1)
    # (2n+1) for shifted_score_diff (representing score_diff from -n to n; shifted by +n)
    
    # Use list comprehension for initialization. All values start at 0.
    dp = [[[0] * (2 * n + 1) for _ in range(4)] for _ in range(n + 1)]

    # Helper function to determine the point difference in a round.
    # Returns (points for Alice, points for Bob)
    def get_round_outcome(A_idx, B_idx):
        """ Determines the points awarded in a round based on Alice's (A_idx) and Bob's (B_idx) moves. """
        if A_idx == B_idx: return (0, 0) # Draw, no points awarded
        # Winning rules: F(0) beats E(2), W(1) beats F(0), E(2) beats W(1)
        # This corresponds to the rule: idx beats (idx + 2) % 3
        
        # Check if Alice wins: Alice's move beats Bob's move
        if A_idx == (B_idx + 2) % 3: return (1, 0) # Alice gets 1 point
        # Otherwise Bob must win: Bob's move beats Alice's move
        else: 
            # Implicitly, B_idx == (A_idx + 2) % 3 holds here.
            return (0, 1) # Bob gets 1 point

    # Base case: At state i = n (after all n rounds have been played)
    # If the final score difference (Bob - Alice) is positive, this path represents one winning sequence for Bob.
    # We mark such states in the DP table with 1.
    for last_B_idx_shifted in range(4): # Iterate through all possible last moves (-1 to 2, shifted to 0 to 3)
        for shifted_score_diff in range(2 * n + 1): # Iterate through all possible final shifted score differences (0 to 2n)
            current_score_diff = shifted_score_diff - n # Convert shifted score diff back to actual score diff (-n to n)
            if current_score_diff > 0:
                # If Bob's score is strictly greater than Alice's score
                dp[n][last_B_idx_shifted][shifted_score_diff] = 1
            # Otherwise, dp[n][...][...] remains 0, indicating this path doesn't lead to a win for Bob.

    # Fill DP table iteratively backwards from round i = n-1 down to 0
    for i in range(n - 1, -1, -1):
        A_idx = char_to_idx[s[i]] # Alice's move in the current round i
        
        # Iterate through all possible states for round i
        for last_B_idx_shifted in range(4): # Represents Bob's move in round i-1 (shifted index 0 to 3)
            last_B_idx = last_B_idx_shifted - 1 # Actual index of Bob's move in round i-1 (-1 means no previous move, i.e. i=0)

            for current_shifted_score_diff in range(2 * n + 1): # Represents score diff up to round i-1 (shifted index 0 to 2n)
                current_score_diff = current_shifted_score_diff - n # Actual score diff
                
                # Calculate the number of winning sequences starting from this state
                count = 0
                
                # Iterate through Bob's possible moves for the current round i (B_idx: 0, 1, 2)
                for B_idx in range(3):
                    # Apply constraint: Bob cannot make the same move as in the previous round (i-1)
                    if B_idx == last_B_idx:
                         continue # Skip this move if it's the same as the last one

                    # Calculate the outcome of round i
                    delta_SA, delta_SB = get_round_outcome(A_idx, B_idx)
                    # Calculate the score difference after round i
                    new_score_diff = current_score_diff + (delta_SB - delta_SA)
                    # Calculate the corresponding shifted score difference index for the DP table
                    new_shifted_score_diff = new_score_diff + n
                    
                    # Check if the new score difference is within the valid range [-n, n] (or shifted [0, 2n])
                    # This check ensures we access the DP table with valid indices. 
                    # It should always be true if logic is correct, as score difference cannot exceed +/- n.
                    if 0 <= new_shifted_score_diff <= 2 * n:
                        # Add the count from the next state (i+1).
                        # Bob's move B_idx in round 'i' becomes the 'last move' for round 'i+1'.
                        # The shifted index for this last move is B_idx + 1.
                        count = (count + dp[i + 1][B_idx + 1][new_shifted_score_diff]) % MOD
                    # else: The score difference calculation resulted in an out-of-bounds value. 
                    # This indicates either a potential logic error or an edge case not handled.
                    # Assuming correct logic, this branch shouldn't be reached. Contribution is 0.
                
                # Store the computed total count for the current state (i, last_B_idx, current_score_diff)
                dp[i][last_B_idx_shifted][current_shifted_score_diff] = count

    # The final answer is the number of ways starting from the initial state:
    # Round i=0, no previous move (last_B_idx = -1, corresponding shifted index 0),
    # and initial score difference 0 (corresponding shifted index n).
    return dp[0][0][n]