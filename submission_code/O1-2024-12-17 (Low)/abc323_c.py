def main():
    import sys

    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    A = list(map(int, data[2:2+M]))
    strings = data[2+M:]
    
    # Precompute the current total scores for each player
    # plus their personal bonus (i points for the i-th player).
    current_scores = []
    for i in range(N):
        solved_score = 0
        for j in range(M):
            if strings[i][j] == 'o':
                solved_score += A[j]
        # Add the player's bonus of i+1 points
        current_scores.append(solved_score + (i+1))
    
    # For each player i, find the minimum number of unsolved problems
    # they must solve such that their total score beats all other players' current scores.
    for i in range(N):
        # The target is one more than the max of other players' scores
        other_max = 0
        for k in range(N):
            if k != i:
                other_max = max(other_max, current_scores[k])
        
        # Gather the scores of unsolved problems for player i
        unsolved_scores = []
        for j in range(M):
            if strings[i][j] == 'x':
                unsolved_scores.append(A[j])
        # Sort in descending order so we pick the largest first
        unsolved_scores.sort(reverse=True)
        
        # Greedily pick scores until we exceed other_max
        base_score = current_scores[i]  # player's current total
        count = 0
        for score in unsolved_scores:
            if base_score > other_max:
                break
            base_score += score
            count += 1
        
        # If still not exceeded at the end, it must exceed after solving all
        # (the problem statement guarantees it is possible).
        if base_score <= other_max:
            count = len(unsolved_scores)
        
        print(count)

# do not forget to call main
if __name__ == "__main__":
    main()