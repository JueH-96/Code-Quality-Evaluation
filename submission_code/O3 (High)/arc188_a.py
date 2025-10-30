import sys
from collections import defaultdict

MOD = 998244353

# toggle value for every concrete character
TOGGLE = {'A': 1, 'B': 2, 'C': 4}

# class id for parity value 0..7   (0: {0,7}, 1:{1,6}, 2:{2,5}, 3:{3,4})
CLASS_ID = [min(v, 7 - v) for v in range(8)]  # 0,1,2,3,3,2,1,0

# pre–compute n choose 2 for n up to 51
CH2 = [n * (n - 1) // 2 for n in range(52)]


def main() -> None:
    N, K = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()

    # DP :  (state, n0, n1, n2)  ->  ways
    dp = {(0, 1, 0, 0): 1}        # before the first character

    for pos, ch in enumerate(S, 1):
        nxt = defaultdict(int)
        if ch == '?':
            toggles = (1, 2, 4)          # A, B, C
        else:
            toggles = (TOGGLE[ch],)

        for (state, n0, n1, n2), ways in dp.items():
            for tg in toggles:
                state2 = state ^ tg
                cls = CLASS_ID[state2]

                if cls == 0:
                    key = (state2, n0 + 1, n1, n2)
                elif cls == 1:
                    key = (state2, n0, n1 + 1, n2)
                elif cls == 2:
                    key = (state2, n0, n1, n2 + 1)
                else:                       # cls == 3
                    key = (state2, n0, n1, n2)

                nxt[key] = (nxt[key] + ways) % MOD
        dp = nxt

    answer = 0
    total_pref = N + 1                     # prefixes including the empty one

    for (state, n0, n1, n2), ways in dp.items():
        n3 = total_pref - (n0 + n1 + n2)
        good = CH2[n0] + CH2[n1] + CH2[n2] + CH2[n3]
        if good >= K:
            answer = (answer + ways) % MOD

    print(answer % MOD)


if __name__ == '__main__':
    main()