import sys
import collections

def main():
    # Read N and T from the first line of input
    N, T = map(int, sys.stdin.readline().split())

    # player_scores[i] stores the current score of player i (0-indexed)
    # All players start with 0 points.
    player_scores = [0] * N  
    
    # score_counts is a Counter that maps a score value to the number of players
    # currently having that score.
    score_counts = collections.Counter()
    # Initially, all N players have a score of 0.
    score_counts[0] = N 
    
    # num_distinct_scores tracks the current number of unique score values
    # present among all players.
    # Since N >= 1 (from problem constraints), initially there's one distinct score (0).
    num_distinct_scores = 1
    
    # output_lines will store the result for each time step.
    # We'll join and print them at the end for efficiency.
    output_lines = []

    # Process T score updates
    for _ in range(T):
        # Read player A_i (1-indexed) and points B_i for the current update
        A_i, B_i = map(int, sys.stdin.readline().split())
        
        player_idx = A_i - 1 # Convert player ID to 0-indexed

        # Get the player's score before this update
        old_score = player_scores[player_idx]
        
        # Calculate the player's new score
        # B_i >= 1, so new_score will always be different from old_score
        new_score = old_score + B_i
        
        # Update the player's score in our tracking array
        player_scores[player_idx] = new_score
        
        # --- Update score_counts and num_distinct_scores ---
        
        # Process the old_score:
        # One less player now has old_score.
        score_counts[old_score] -= 1
        # If the count of old_score drops to 0, it means this score value
        # is no longer possessed by any player. So, decrement num_distinct_scores.
        if score_counts[old_score] == 0:
            num_distinct_scores -= 1
        
        # Process the new_score:
        # Before incrementing the count for new_score, check its current count.
        # If score_counts[new_score] is 0, it means new_score was not among the
        # distinct scores (or its count had previously dropped to 0).
        # So, its appearance (or reappearance) increases num_distinct_scores.
        if score_counts[new_score] == 0:
            num_distinct_scores += 1
        # One more player now has new_score.
        score_counts[new_score] += 1
        
        # Add the current number of distinct scores to our list of results
        output_lines.append(str(num_distinct_scores))

    # Print all results, each on a new line
    sys.stdout.write("
".join(output_lines) + "
")

if __name__ == '__main__':
    main()