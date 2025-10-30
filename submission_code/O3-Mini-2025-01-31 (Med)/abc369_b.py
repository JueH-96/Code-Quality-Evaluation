def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return

    it = iter(input_data)
    N = int(next(it))
    moves = []
    for _ in range(N):
        key = int(next(it))
        hand = next(it)
        moves.append((key, hand))
    
    INF = 10**9
    # dp[i][l][r] will store the minimum fatigue level after i key presses
    # with the left hand on key l and the right hand on key r.
    # Keys range from 1 to 100, so we use indices 1..100.
    dp = [[[INF] * 101 for _ in range(101)] for _ in range(N + 1)]
    
    # Before any press, we can freely position both hands,
    # so we initialize all dp[0][l][r] = 0 for l, r in 1..100.
    for l in range(1, 101):
        for r in range(1, 101):
            dp[0][l][r] = 0

    # Process each key press sequentially.
    for i in range(N):
        key, hand = moves[i]
        for l in range(1, 101):
            for r in range(1, 101):
                current_cost = dp[i][l][r]
                if current_cost == INF:
                    continue

                if hand == 'L':
                    # To press with the left hand, we move it from position l to key.
                    new_cost = current_cost + abs(key - l)
                    # The left hand is updated to key; the right hand stays at r.
                    if new_cost < dp[i+1][key][r]:
                        dp[i+1][key][r] = new_cost
                else:  # hand == 'R'
                    new_cost = current_cost + abs(key - r)
                    # The right hand is updated to key; the left hand remains at l.
                    if new_cost < dp[i+1][l][key]:
                        dp[i+1][l][key] = new_cost

    # After all key presses, the answer is the minimum fatigue over all possible positions.
    answer = INF
    for l in range(1, 101):
        for r in range(1, 101):
            answer = min(answer, dp[N][l][r])
    
    sys.stdout.write(str(answer))

if __name__ == '__main__':
    main()