def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    X = int(data[1])
    A = list(map(int, data[2:]))

    # We'll try all possible scores for the N-th round from 0 to 100.
    # For each candidate score, insert it into the existing scores,
    # sort them, remove the highest and lowest, and check the sum.
    # The first (smallest) candidate that achieves a final grade >= X
    # will be our answer. If none do, return -1.
    
    for candidate_score in range(101):  # 0 through 100 inclusive
        scores_with_candidate = A + [candidate_score]
        sorted_scores = sorted(scores_with_candidate)
        final_grade = sum(sorted_scores[1:-1])  # sum of N-2 middle scores
        if final_grade >= X:
            print(candidate_score)
            return
    
    # If we reach here, no candidate score from 0 to 100 yielded final_grade >= X
    print(-1)

# Do not forget to call main().
if __name__ == "__main__":
    main()