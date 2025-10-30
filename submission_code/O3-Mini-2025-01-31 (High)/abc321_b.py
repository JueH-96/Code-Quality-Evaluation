def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return

    N = int(data[0])
    X = int(data[1])
    # There are N-1 scores given for rounds 1 to N-1.
    scores = list(map(int, data[2:2 + N - 1]))

    # Try every candidate score x for round N from 0 to 100.
    for x in range(101):
        # Combine current scores with candidate score x.
        current_scores = scores + [x]
        current_scores.sort()
        # The final grade is the sum of scores excluding the smallest and largest.
        final_grade = sum(current_scores[1:-1])
        if final_grade >= X:
            print(x)
            return

    # If no candidate x in 0..100 gives final_grade >= X, output -1.
    print(-1)

if __name__ == "__main__":
    main()