import sys


def main() -> None:
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()

    # Encode moves to integers: 0 = R, 1 = P, 2 = S
    to_id = {'R': 0, 'P': 1, 'S': 2}

    # win_map[m] = move beaten by m
    win_map = {0: 2, 1: 0, 2: 1}   # R beats S, P beats R, S beats P

    NEG = -10 ** 9
    dp = [NEG, NEG, NEG]           # dp[last_move] = maximum wins up to current position

    for i, ch in enumerate(S):
        a = to_id[ch]              # Aoki's move as id
        new_dp = [NEG, NEG, NEG]

        for m in range(3):         # Takahashi's move at this position
            # Move m is allowed if it is a draw or a win (Takahashi must not lose)
            if m == a or win_map[m] == a:
                gain = 1 if win_map[m] == a else 0  # 1 if this move wins, 0 if it draws

                if i == 0:          # First game: no previous move
                    new_dp[m] = gain
                else:
                    for prev in range(3):
                        if prev != m and dp[prev] != NEG:
                            cand = dp[prev] + gain
                            if cand > new_dp[m]:
                                new_dp[m] = cand

        dp = new_dp

    print(max(dp))


if __name__ == "__main__":
    main()