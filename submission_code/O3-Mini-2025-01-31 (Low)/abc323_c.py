def main():
    import sys
    input_data = sys.stdin.read().split()
    # read inputs
    it = iter(input_data)
    N = int(next(it))
    M = int(next(it))
    scores = [int(next(it)) for _ in range(M)]
    solved = [next(it).strip() for _ in range(N)]
    
    # Calculate each player's current total score (bonus + solved problems)
    current_scores = [0] * N
    for i in range(N):
        bonus = i + 1  # bonus is player's index (1-indexed)
        total = bonus
        for j in range(M):
            if solved[i][j] == 'o':
                total += scores[j]
        current_scores[i] = total

    # For each player i, compute the minimum number of additional problems needed.
    # Player i's unsolved problem scores, sorted descending.
    # They need to beat the highest current score among all opponents.
    result_lines = []
    for i in range(N):
        # Find the highest score among opponents.
        max_opponent_score = 0
        for j in range(N):
            if i == j:
                continue
            if current_scores[j] > max_opponent_score:
                max_opponent_score = current_scores[j]
        
        # If player i is already ahead, then answer is 0.
        if current_scores[i] > max_opponent_score:
            result_lines.append("0")
            continue
        
        # Compute the extra score required: need to be strictly greater.
        required_extra = (max_opponent_score - current_scores[i]) + 1
        
        # Gather scores of unsolved problems.
        unsolved_scores = []
        for j in range(M):
            if solved[i][j] == 'x':
                unsolved_scores.append(scores[j])
        # Sort in descending order for greedily choosing highest scoring problems.
        unsolved_scores.sort(reverse=True)
        
        # Greedily pick problems until surpassing the required extra score.
        count = 0
        accumulated = 0
        for score in unsolved_scores:
            accumulated += score
            count += 1
            if accumulated >= required_extra:
                break
        result_lines.append(str(count))
    
    # Output the result for each player.
    sys.stdout.write("
".join(result_lines))
    
if __name__ == '__main__':
    main()