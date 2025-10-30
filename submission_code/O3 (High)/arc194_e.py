import sys
import math


def has_run(s: str, ch: str, length: int) -> bool:
    """whether s contains `ch` repeated at least `length` times consecutively"""
    if length == 0:          # X,Y are positive so this is never used, but keep safe
        return True
    cur = 0
    for c in s:
        if c == ch:
            cur += 1
            if cur >= length:
                return True
        else:
            cur = 0
    return False


def inversion_mod(s: str, mod: int) -> int:
    """
    number of pairs (i<j, s[i]='1', s[j]='0')  taken modulo `mod`
    if mod == 1 the value is always 0 so we can skip the work
    """
    if mod == 1:
        return 0
    inv = 0
    ones = 0
    for c in s:
        if c == '1':
            ones += 1
        else:  # c == '0'
            inv += ones
            if inv >= mod:          # keep value small
                inv %= mod
    return inv % mod


def main() -> None:
    it = sys.stdin.read().strip().split()
    if not it:
        return
    N, X, Y = map(int, it[:3])
    S = it[3]
    T = it[4]

    # already equal
    if S == T:
        print("Yes")
        return

    window = X + Y

    # no operation fits inside the string
    if N < window:
        print("No")
        return

    # the window covers the whole string: only two states are reachable
    if N == window:
        reachable = {S}
        if S[:X] == '0' * X and S[X:] == '1' * Y:
            reachable.add('1' * Y + '0' * X)
        elif S[:Y] == '1' * Y and S[Y:] == '0' * X:
            reachable.add('0' * X + '1' * Y)
        print("Yes" if T in reachable else "No")
        return

    # ----- from here we have many possible positions for the window -----

    # global counts have to match
    if S.count('0') != T.count('0'):
        print("No")
        return

    # we must be able to perform at least one operation, and the
    # 'existence of a run' is preserved by every operation
    if not has_run(S, '0', X) or not has_run(S, '1', Y):
        print("No")
        return
    if not has_run(T, '0', X) or not has_run(T, '1', Y):
        print("No")
        return

    # characters never change residue class modulo gcd(X, Y)
    g = math.gcd(X, Y)
    cnt_s = [0] * g
    cnt_t = [0] * g
    for i, ch in enumerate(S):
        if ch == '0':
            cnt_s[i % g] += 1
    for i, ch in enumerate(T):
        if ch == '0':
            cnt_t[i % g] += 1
    if cnt_s != cnt_t:
        print("No")
        return

    # the number of inversions is preserved modulo X*Y
    m = X * Y
    if inversion_mod(S, m) != inversion_mod(T, m):
        print("No")
        return

    print("Yes")


if __name__ == "__main__":
    main()