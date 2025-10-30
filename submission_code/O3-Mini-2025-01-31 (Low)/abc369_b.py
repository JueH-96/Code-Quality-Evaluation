def main():
    import sys
    input = sys.stdin.readline

    N = int(input())
    moves = []
    for _ in range(N):
        a, s = input().split()
        moves.append((int(a), s))
    
    # dp[i][l][r] : minimum fatigue cost after processing i moves
    # with left hand on key l and right hand on key r.
    # Note: keys are numbered from 1 to 100.
    INF = 10**9
    dp = [[[INF] * 101 for _ in range(101)] for _ in range(N + 1)]
    
    # Initially, both hands can be placed anywhere with 0 cost.
    # For every possible starting positions (l, r) set cost = 0.
    for l in range(1, 101):
        for r in range(1, 101):
            dp[0][l][r] = 0
    
    # Process each move.
    # When a move instructs to use the left hand, the cost is |A - current_left|.
    # Similarly for the right hand.
    for i in range(N):
        A, hand = moves[i]
        for l in range(1, 101):
            for r in range(1, 101):
                current = dp[i][l][r]
                if current == INF:
                    continue
                if hand == 'L':
                    # Move left hand from l to A, cost = |A - l|.
                    new_cost = current + abs(A - l)
                    # Press with left hand giving new left key = A; right hand remains at r.
                    if new_cost < dp[i + 1][A][r]:
                        dp[i + 1][A][r] = new_cost
                else:  # hand == 'R'
                    # Move right hand from r to A, cost = |A - r|.
                    new_cost = current + abs(A - r)
                    # Press with right hand giving new right key = A; left hand remains at l.
                    if new_cost < dp[i + 1][l][A]:
                        dp[i + 1][l][A] = new_cost

    # The answer is the minimal fatigue after N moves over all possible hand positions.
    answer = INF
    for l in range(1, 101):
        for r in range(1, 101):
            answer = min(answer, dp[N][l][r])
    
    sys.stdout.write(str(answer))


if __name__ == "__main__":
    main()