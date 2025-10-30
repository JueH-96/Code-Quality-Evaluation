import sys

def main():
    input = sys.stdin.readline
    N = int(input())
    S = input().strip()
    # Map R, P, S to 0,1,2
    mp = {'R': 0, 'P': 1, 'S': 2}
    # We use a large negative for impossible states
    NEG_INF = -10**18

    # dp_prev[m] = max wins up to previous game, ending with move m
    dp_prev = [NEG_INF] * 3

    # Initialize for game 0
    s0 = mp[S[0]]
    # Allowed moves: tie (m == s0) or win (m == (s0+1)%3)
    for m in (s0, (s0 + 1) % 3):
        dp_prev[m] = 1 if m == (s0 + 1) % 3 else 0

    # Iterate games 1..N-1
    for i in range(1, N):
        si = mp[S[i]]
        tie_move = si
        win_move = (si + 1) % 3

        dp_curr = [NEG_INF] * 3

        # For tie move (no win)
        m = tie_move
        best = NEG_INF
        # pick best from dp_prev[m'] where m' != m
        for m_prev in range(3):
            if m_prev != m and dp_prev[m_prev] > best:
                best = dp_prev[m_prev]
        dp_curr[m] = best + 0

        # For win move (counts as a win)
        m = win_move
        best = NEG_INF
        for m_prev in range(3):
            if m_prev != m and dp_prev[m_prev] > best:
                best = dp_prev[m_prev]
        dp_curr[m] = best + 1

        dp_prev = dp_curr

    # Answer is the best of the last game's dp
    print(max(dp_prev))


if __name__ == "__main__":
    main()