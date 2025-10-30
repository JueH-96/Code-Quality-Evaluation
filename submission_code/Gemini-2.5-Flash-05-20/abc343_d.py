import sys
from collections import Counter

def solve():
    # Read N (number of players) and T (number of events) from the first line
    N, T = map(int, sys.stdin.readline().split())

    # Initialize player_scores array. player_scores[i] will store the current score of player (i+1).
    # All players start with 0 points.
    player_scores = [0] * N

    # Initialize score_counts using collections.Counter.
    # This Counter will store the frequency of each score value currently held by players.
    # Initially, N players have a score of 0.
    score_counts = Counter({0: N})

    # Iterate through T events
    for _ in range(T):
        # Read the player index (A_i) and points (B_i) for the current event
        A_i, B_i = map(int, sys.stdin.readline().split())
        
        # Adjust A_i to be 0-indexed for array access (player 1 is at index 0, player N at index N-1)
        player_idx = A_i - 1

        # Get the current score of the player whose score is about to change
        old_score = player_scores[player_idx]

        # Decrease the count of the old score in our frequency map.
        # Counter automatically handles if old_score was not explicitly present (though it would be in this logic).
        score_counts[old_score] -= 1
        
        # If the count of old_score drops to 0, Counter will automatically treat it as "not present"
        # for operations like len(), which means it won't contribute to the distinct count.
        # No explicit `del score_counts[old_score]` is needed for len() to work correctly.

        # Calculate the new score for the player
        new_score = old_score + B_i

        # Update the player's score in the player_scores array
        player_scores[player_idx] = new_score

        # Increase the count of the new score in our frequency map.
        # If new_score didn't exist, it will be added with count 1.
        # If it existed, its count will be incremented.
        score_counts[new_score] += 1

        # The number of different score values is simply the number of unique keys
        # in the score_counts Counter that have a non-zero count.
        # `len(score_counts)` provides this directly and efficiently.
        sys.stdout.write(str(len(score_counts)) + "
")

# Call the solve function to execute the program
if __name__ == '__main__':
    solve()