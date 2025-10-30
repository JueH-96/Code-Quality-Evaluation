def main():
    """
    Reads the problem input, solves it, and prints the output.
    """
    # Read N (total number of rounds) and X (target final grade).
    try:
        N, X = map(int, input().split())
        # Read the list of N-1 scores from the completed rounds.
        A = list(map(int, input().split()))
    except (IOError, ValueError):
        # This handles cases of malformed input, though not expected in a contest setting.
        return

    # Iterate through all possible integer scores for the N-th round (0 to 100).
    for last_round_score in range(101):
        # Create the complete list of N scores.
        all_scores = A + [last_round_score]
        
        # Sort the scores in ascending order to easily find the min, max, and middle values.
        all_scores.sort()
        
        # The final grade is the sum of the N-2 scores excluding the highest and lowest.
        # Python's list slice `[1:-1]` selects all elements except the first and last.
        final_grade = sum(all_scores[1:-1])

        # Check if the final grade meets or exceeds the target X.
        if final_grade >= X:
            # Since we are iterating from 0 upwards, the first score that works
            # is the minimum required score.
            print(last_round_score)
            return

    # If the loop finishes without finding a valid score, it's impossible.
    print(-1)

if __name__ == "__main__":
    main()