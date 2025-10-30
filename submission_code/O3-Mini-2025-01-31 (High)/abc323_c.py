def main():
    import sys
    data = sys.stdin.read().strip().splitlines()
    if not data:
        return
    # Read N and M.
    parts = data[0].split()
    N = int(parts[0])
    M = int(parts[1])
    
    # Read problem scores.
    A = list(map(int, data[1].split()))
    
    # Read each player's solved pattern.
    S_list = [line.strip() for line in data[2:2+N]]
    
    # Compute current scores for each player.
    # Each player's score is: bonus (player number i, 1-indexed) + sum of scores of solved problems.
    current_scores = []
    for i in range(N):
        s = S_list[i]
        total = i + 1  # bonus points (1-indexed).
        for j in range(M):
            if s[j] == 'o':
                total += A[j]
        current_scores.append(total)
    
    # For each player, determine the minimum number of additional problems (from unsolved ones) 
    # needed to exceed all other players' current scores.
    results = []
    for i in range(N):
        # Find the maximum score among all opponents.
        opp_max = max(current_scores[j] for j in range(N) if j != i)
        # If the player's current score is already larger than every opponent's score,
        # then no additional problems are needed.
        if current_scores[i] > opp_max:
            results.append(0)
        else:
            # The player's new score must be at least opp_max+1.
            needed = opp_max + 1 - current_scores[i]
            # Get the list of unsolved problems scores.
            unsolved = [A[j] for j in range(M) if S_list[i][j] == 'x']
            # To minimize the number of problems to solve, sort unsolved problem scores in descending order.
            unsolved.sort(reverse=True)
            acc = 0
            cnt = 0
            for score in unsolved:
                acc += score
                cnt += 1
                if acc >= needed:
                    break
            results.append(cnt)
    
    # Output the results: one answer per line.
    sys.stdout.write("
".join(map(str, results)))

if __name__ == '__main__':
    main()