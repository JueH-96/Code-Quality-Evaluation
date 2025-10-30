def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, X = map(int, data[:2])
    A = list(map(int, data[2:]))

    # We'll try each possible score for the N-th round from 0 to 100 inclusive.
    for possible_score in range(101):
        scores = A + [possible_score]
        scores.sort()
        # The final grade is the sum of all scores except the highest and the lowest.
        final_grade = sum(scores[1:-1])
        if final_grade >= X:
            print(possible_score)
            return
    
    # If we cannot reach the target X for any possible score, print -1.
    print(-1)

# Do not forget to call the main function
if __name__ == "__main__":
    main()