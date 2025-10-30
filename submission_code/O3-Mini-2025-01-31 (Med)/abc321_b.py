def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    # Parse inputs: first value is N, second is X, followed by N-1 scores
    N = int(data[0])
    X = int(data[1])
    scores_given = list(map(int, data[2:2 + N - 1]))
    
    # We try each possible score for round N from 0 to 100 and check if we can meet or exceed the target X.
    for round_N_score in range(101):
        # Combine the given scores with the candidate score for round N.
        all_scores = scores_given + [round_N_score]
        # Sort the scores in ascending order.
        sorted_scores = sorted(all_scores)
        # The final grade is the sum of scores excluding the lowest and highest.
        final_grade = sum(sorted_scores[1:-1])
        # If the final grade meets or exceeds X, we print the candidate score and return.
        if final_grade >= X:
            sys.stdout.write(str(round_N_score))
            return
    # If no candidate score from 0 to 100 can yield at least X, print -1.
    sys.stdout.write("-1")

if __name__ == '__main__':
    main()