def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N, X = map(int, data[:2])
    A = list(map(int, data[2:]))

    # We need to find the minimum score in [0..100] for round N that yields
    # a final grade (sum of middle N-2 scores) >= X.

    best_score = None
    for candidate_score in range(101):
        all_scores = A + [candidate_score]
        all_scores.sort()
        # Exclude the lowest and highest score
        middle_sum = sum(all_scores[1:-1])
        if middle_sum >= X:
            best_score = candidate_score
            break

    # If no score in [0..100] yields at least X, print -1
    if best_score is None:
        print(-1)
    else:
        print(best_score)

def main():
    solve()

if __name__ == "__main__":
    main()