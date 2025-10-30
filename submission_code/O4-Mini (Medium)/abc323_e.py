import sys
import threading
def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0]); X = int(data[1])
    T = list(map(int, data[2:] ))
    mod = 998244353

    invN = pow(N, mod-2, mod)

    # f[s] = probability (mod) that a song starts exactly at time s
    # f[0] = 1, transitions f[s + T_i] += f[s] * (1/N)
    f = [0] * (X+1)
    f[0] = 1

    # Only keep T_i <= X for transitions
    # But we still need T1 separately
    T1 = T[0]
    Ts = [t for t in T if t <= X]

    for s in range(0, X+1):
        fs = f[s]
        if fs:
            # distribute to s + t
            add = fs * invN % mod
            # loop only Ts
            for t in Ts:
                st = s + t
                if st > X:
                    # since Ts sorted? We didn't sort, but break seldom happens
                    continue
                f[st] = (f[st] + add) % mod

    # we want sum_{s = max(0, X-T1+1) .. X} f[s] * (1/N)
    lo = X - T1 + 1
    if lo < 0:
        lo = 0
    total = 0
    # sum f[s]
    for s in range(lo, X+1):
        total = (total + f[s]) % mod

    ans = total * invN % mod
    print(ans)

if __name__ == "__main__":
    main()