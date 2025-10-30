def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    it = iter(input_data)
    N = int(next(it))
    T = int(next(it))
    
    # Initialize scores: all players start at 0.
    scores = [0] * N
    # Use a dictionary to track how many players have a specific score.
    score_counts = {0: N}
    
    results = []
    for _ in range(T):
        # Read update event
        player = int(next(it)) - 1  # Convert to 0-indexed.
        add_points = int(next(it))
        
        # Get the player's current score and update it.
        current_score = scores[player]
        new_score = current_score + add_points
        scores[player] = new_score
        
        # Decrement the count for the old score.
        if score_counts[current_score] == 1:
            del score_counts[current_score]
        else:
            score_counts[current_score] -= 1
        
        # Increment the count for the new score.
        score_counts[new_score] = score_counts.get(new_score, 0) + 1
        
        # Record the number of distinct scores.
        results.append(str(len(score_counts)))
    
    sys.stdout.write("
".join(results))

if __name__ == '__main__':
    main()