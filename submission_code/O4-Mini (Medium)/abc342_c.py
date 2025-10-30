import sys
import threading

def main():
    data = sys.stdin.read().split()
    # data layout: N, S, Q, then 2*Q letters
    it = iter(data)
    N = int(next(it))
    S = next(it)
    Q = int(next(it))
    # initialize mapping f: f[x] = current mapping of character x (0=a,1=b,...)
    f = list(range(26))
    for _ in range(Q):
        c = next(it)
        d = next(it)
        ci = ord(c) - ord('a')
        di = ord(d) - ord('a')
        # replace all mappings equal to ci with di
        # i.e., for every original letter x, if f[x] == ci, set f[x] = di
        for x in range(26):
            if f[x] == ci:
                f[x] = di
    # build resulting string
    res = []
    oa = ord('a')
    for ch in S:
        idx = ord(ch) - oa
        res.append(chr(f[idx] + oa))
    sys.stdout.write(''.join(res))

if __name__ == "__main__":
    main()