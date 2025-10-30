import sys

def solve():
    # Read N and T
    line = sys.stdin.readline().split()
    N = int(line[0])
    T = int(line[1])

    # Initialize scores for N players to 0
    scores = [0] * N

    # Initialize dictionary to count scores. All N players start with score 0.
    # Constraints state N >= 1, so score 0 is initially present.
    score_counts = {0: N}

    # Process T updates
    for _ in range(T):
        # Read the update
        line = sys.stdin.readline().split()
        A = int(line[0])
        B = int(line[1])
        
        # Player index (0-based)
        player_idx = A - 1

        # Get the current score of the player
        current_score = scores[player_idx]

        # Calculate the new score
        new_score = current_score + B

        # --- Update score_counts ---

        # Decrement the count for the current score
        # This score must exist in score_counts because the player had this score.
        score_counts[current_score] -= 1

        # If the count for the current score becomes 0,
        # it means this score value is no longer held by any player.
        # Remove the score from the dictionary
        if score_counts[current_score] == 0:
            del score_counts[current_score]

        # Increment the count for the new score
        # If the new score is not yet in the dictionary, get() will return 0.
        # Otherwise, get() returns the current count. We add 1 in either case.
        score_counts[new_score] = score_counts.get(new_score, 0) + 1

        # --- Update the player's score in the scores array ---
        scores[player_idx] = new_score

        # The number of distinct scores is simply the number of keys in the dictionary
        distinct_count = len(score_counts)

        # Print the number of distinct scores after this update
        sys.stdout.write(str(distinct_count) + '
')

solve()