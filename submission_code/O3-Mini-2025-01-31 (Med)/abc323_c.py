def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    problems = [int(next(it)) for _ in range(M)]
    # Read the solved/unsolved info for each player.
    players = [next(it).strip() for _ in range(N)]
    
    # Calculate the current total score for each player:
    # Sum of scores for solved problems + bonus equal to player's number (1-indexed).
    current_scores = []
    for i in range(N):
        s = players[i]
        score_sum = 0
        for j, ch in enumerate(s):
            if ch == 'o':
                score_sum += problems[j]
        current_scores.append(score_sum + (i + 1))
    
    # For each player, determine the minimum number of unsolved problems needed
    # such that after solving them (choosing problems with the highest
    # scores first), the player's new total score exceeds all other players' current scores.
    results = []
    for i in range(N):
        curr = current_scores[i]
        # Find the maximum score among all other players.
        threshold = max(current_scores[j] for j in range(N) if j != i)
        
        # If already ahead, no additional problems are needed.
        if curr > threshold:
            results.append("0")
            continue
        
        # Gather candidate problems this player has not solved.
        # Solving the highest valued unsolved problems minimizes the number required.
        unsolved_scores = []
        s = players[i]
        for j, ch in enumerate(s):
            if ch == 'x':
                unsolved_scores.append(problems[j])
        unsolved_scores.sort(reverse=True)
        
        # Greedily select problems until the new total score exceeds the threshold.
        additional = 0
        count = 0
        for score in unsolved_scores:
            additional += score
            count += 1
            if curr + additional > threshold:
                break
        results.append(str(count))
    
    sys.stdout.write("
".join(results))
    
if __name__ == '__main__':
    main()