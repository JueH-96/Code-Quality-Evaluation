import sys

def solve():
    S, N = sys.stdin.read().split()
    N = int(N)
    T = set()
    for i in range(2**S.count('?')):
        bin_str = bin(i)[2:].zfill(S.count('?'))
        t = S
        for b in bin_str:
            t = t.replace('?', b, 1)
        T.add(int(t, 2))
    T = [i for i in T if i <= N]
    if T:
        print(max(T))
    else:
        print(-1)

solve()