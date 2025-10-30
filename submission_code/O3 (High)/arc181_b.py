import sys, math


def is_periodic(s: str, d: int) -> bool:
    """
    Return True iff `s` has period `d`
    (i.e. s[i] == s[i-d] for every i ≥ d).
    By convention d == 0 or d == len(s) is always a period.
    """
    n = len(s)
    if d == 0 or d == n:
        return True
    # fast test: first and last char must match
    if s[0] != s[-d]:
        return False
    for i in range(d, n):
        if s[i] != s[i - d]:
            return False
    return True


def solve_case(S: str, X: str, Y: str) -> str:
    ls = len(S)

    x0 = X.count('0')
    x1 = len(X) - x0
    y0 = Y.count('0')
    y1 = len(Y) - y0

    diff1 = x1 - y1           #  (#1 in X) - (#1 in Y)
    diff0 = y0 - x0           #  (#0 in Y) - (#0 in X)

    # |X0|*|S| + |X1|*|T| = |Y0|*|S| + |Y1|*|T|
    #  diff1 * |T| = diff0 * |S|

    if diff1 == 0:              # left hand side is always 0
        return 'Yes' if diff0 == 0 else 'No'

    numerator = diff0 * ls
    if numerator % diff1 != 0:          # length must be integer
        return 'No'

    L = numerator // diff1              # candidate |T|
    if L < 0:                           # length cannot be negative
        return 'No'
    if L == 0:                          # empty string always works
        return 'Yes'

    d = math.gcd(ls, L)                 # common length of potential base word
    return 'Yes' if is_periodic(S, d) else 'No'


def main() -> None:
    data = sys.stdin.buffer.read().splitlines()
    t = int(data[0])
    out = []
    idx = 1
    for _ in range(t):
        S = data[idx].decode().rstrip(); idx += 1
        X = data[idx].decode().rstrip(); idx += 1
        Y = data[idx].decode().rstrip(); idx += 1
        out.append(solve_case(S, X, Y))
    sys.stdout.write('
'.join(out))


if __name__ == '__main__':
    main()