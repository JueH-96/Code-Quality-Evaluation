import sys

def main() -> None:
    input = sys.stdin.readline

    N = int(input().strip())
    S = input().strip()
    costs = [0] + list(map(int, input().split()))          # 1-indexed

    # ---------- prefix cost for perfect alternation ----------
    # pref0[i] : cost to make S[1..i] alternate starting with 0
    # pref1[i] : cost to make S[1..i] alternate starting with 1
    pref0 = [0] * (N + 1)
    pref1 = [0] * (N + 1)

    for i in range(1, N + 1):
        bit = int(S[i - 1])

        exp0 = (i - 1) & 1                     # expected bit at i if start with 0
        exp1 = exp0 ^ 1                        # expected bit at i if start with 1

        pref0[i] = pref0[i - 1] + (costs[i] if bit != exp0 else 0)
        pref1[i] = pref1[i - 1] + (costs[i] if bit != exp1 else 0)

    # ---------- suffix cost for alternation with given starting bit ----------
    # f0[i] : cost to make S[i..N] alternate, starting with 0 at position i
    # f1[i] : same but starting with 1 at position i
    f0 = [0] * (N + 2)                     # f? [N+1] = 0 already
    f1 = [0] * (N + 2)

    for i in range(N, 0, -1):
        bit = int(S[i - 1])
        to0 = costs[i] if bit != 0 else 0
        to1 = costs[i] if bit != 1 else 0

        f0[i] = to0 + f1[i + 1]            # next position expects 1
        f1[i] = to1 + f0[i + 1]            # next position expects 0

    # ---------- examine every position of the single equal pair ----------
    ans = float('inf')
    for i in range(1, N):                  # equal pair is (i, i+1)
        parity = (i - 1) & 1               # = 0 if i is odd, 1 if i is even

        # case 1: prefix starts with 0
        v = parity                         # value of T_i and T_{i+1}
        suffix = f0[i + 1] if v == 0 else f1[i + 1]
        ans = min(ans, pref0[i] + suffix)

        # case 2: prefix starts with 1
        v = parity ^ 1
        suffix = f0[i + 1] if v == 0 else f1[i + 1]
        ans = min(ans, pref1[i] + suffix)

    print(ans)

if __name__ == "__main__":
    main()