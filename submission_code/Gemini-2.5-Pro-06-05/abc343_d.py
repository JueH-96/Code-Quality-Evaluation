import sys

def solve():
    """
    Solves the problem by efficiently tracking the number of distinct scores.
    """
    # Use fast I/O for performance with large inputs
    input = sys.stdin.readline
    
    # Read the number of players (N) and the number of updates (T)
    try:
        line = input()
        if not line:
            return
        N, T = map(int, line.split())
    except (IOError, ValueError):
        # Gracefully handle potential empty input lines or parsing errors
        return

    # scores[p] stores the current score of player p (using 1-based indexing).
    # This provides O(1) lookup for a player's score.
    scores = [0] * (N + 1)
    
    # score_counts[s] stores the number of players who currently have score s.
    # This is the core data structure for tracking distinct scores efficiently.
    # Initially, all N players have a score of 0.
    score_counts = {0: N}
    
    # Process all T updates sequentially.
    for _ in range(T):
        # Read the current update: player A's score increases by B.
        A, B = map(int, input().split())

        # --- Step 1: Handle the player's old score ---
        old_score = scores[A]
        
        # The player is moving away from this score, so decrement its count.
        score_counts[old_score] -= 1
        
        # If the count for the old score drops to 0, it means no player
        # has this score anymore. Remove it from the dictionary of distinct scores.
        if score_counts[old_score] == 0:
            del score_counts[old_score]
            
        # --- Step 2: Handle the player's new score ---
        new_score = old_score + B
        
        # Update the player's score in the main tracking list.
        scores[A] = new_score
        
        # The player now has the new score. Increment its count.
        # The .get(key, 0) method is a convenient way to handle cases where
        # the new score is seen for the first time. If new_score is not
        # in the dictionary, .get() returns 0, and the new count becomes 1.
        score_counts[new_score] = score_counts.get(new_score, 0) + 1
        
        # --- Step 3: Report the result for this time step ---
        # The number of distinct scores is the number of keys in score_counts.
        # This is an O(1) operation for dictionaries.
        print(len(score_counts))

# Execute the main logic
solve()