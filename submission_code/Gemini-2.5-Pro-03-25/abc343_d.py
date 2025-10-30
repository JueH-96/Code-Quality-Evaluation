# YOUR CODE HERE
import sys

# Function to contain the logic
def solve():
    # Read N (number of players) and T (number of time steps/updates)
    N, T = map(int, sys.stdin.readline().split())
    
    # Stores the current score of each player. Indexed from 0 to N-1.
    # Initialized to 0 for all players.
    player_scores = [0] * N 
    
    # Dictionary: maps score -> count of players with this score
    # Keys are score values, values are the number of players having that score.
    score_counts = {} 
    
    # Initialize distinct score count and the score_counts dictionary.
    distinct_score_count = 0 
    if N > 0:
       # Initially all N players have score 0.
       score_counts[0] = N 
       # So there is 1 distinct score value (0) initially.
       distinct_score_count = 1

    # Process T updates sequentially
    for _ in range(T):
        # Read player index A_i (1-based) and score increase B_i for this time step
        line = sys.stdin.readline().split()
        A_i = int(line[0]) # Player number (1 to N)
        B_i = int(line[1]) # Points to add
        
        # Convert player index A_i to 0-based index p for list access
        p = A_i - 1 
        
        # --- Update effects related to the player's old score ---
        
        # Get the player's score before the update
        S_old = player_scores[p]
        
        # Since player p is changing score, they no longer have score S_old.
        # Decrement the count of players having score S_old.
        # We are guaranteed that S_old exists as a key in score_counts if its count was positive.
        score_counts[S_old] -= 1
        
        # Check if the count for S_old has dropped to 0.
        # If yes, this score value is no longer present among any players.
        if score_counts[S_old] == 0:
            # The total number of distinct scores decreases by 1.
            distinct_score_count -= 1
            # Remove the score S_old from the dictionary. This keeps the dictionary size manageable
            # and might improve performance slightly.
            del score_counts[S_old] 
            
        # --- Update effects related to the player's new score ---
        
        # Calculate the player's new score after adding B_i points
        S_new = S_old + B_i
        
        # Update the player's score in our tracking list
        player_scores[p] = S_new
        
        # Now consider the impact of player p having the new score S_new.
        # We need to know how many players already had score S_new *before* player p got it.
        # Use dictionary's .get(key, default_value) method. If S_new is not a key, it returns 0.
        current_count_S_new = score_counts.get(S_new, 0)

        # Determine if the score value S_new is newly appearing in the set of scores.
        # This occurs if its count was 0 before player p acquired it.
        # This check must be done *before* we increment the count for S_new.
        is_newly_appearing = (current_count_S_new == 0)

        # Increment the count for S_new, reflecting that player p now has this score.
        # This updates the existing count or adds S_new to the dictionary if it wasn't present.
        score_counts[S_new] = current_count_S_new + 1
        
        # If S_new was newly appearing (its count transitioned from 0 to 1),
        # then the total number of distinct scores increases by 1.
        if is_newly_appearing:
             distinct_score_count += 1
        
        # After processing the update for this time step, print the current total number of distinct scores.
        print(distinct_score_count)

# Call the function to execute the solution logic based on standard input
solve()