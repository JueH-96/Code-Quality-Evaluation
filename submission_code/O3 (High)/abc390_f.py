import sys


def main() -> None:
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    A = [0] + A                       # 1-based

    # positions of every value
    pos = [[] for _ in range(N + 2)]
    for i in range(1, N + 1):
        pos[A[i]].append(i)

    # ---------- part 1 :  Σ distinct ----------
    last = [0] * (N + 2)
    S1 = 0
    for i in range(1, N + 1):
        a = A[i]
        contribution = (i - last[a]) * (N - i + 1)
        S1 += contribution
        last[a] = i

    total_sub = N * (N + 1) // 2

    # ---------- single-value miss counts ----------
    miss = [0] * (N + 2)          # miss[v] = sub arrays not containing v
    for v in range(1, N + 1):
        prev = 0
        s = 0
        for p in pos[v]:
            gap = p - prev - 1
            s += gap * (gap + 1) // 2
            prev = p
        gap = N - prev
        s += gap * (gap + 1) // 2
        miss[v] = s

    # ---------- pair miss counts & ΣE ----------
    S2 = 0
    for v in range(1, N):
        # merge pos[v] and pos[v+1]
        pv, pw = pos[v], pos[v + 1]
        i = j = 0
        prev = 0
        s = 0
        while i < len(pv) or j < len(pw):
            if j == len(pw) or (i < len(pv) and pv[i] < pw[j]):
                cur = pv[i]
                i += 1
            else:
                cur = pw[j]
                j += 1
            gap = cur - prev - 1
            s += gap * (gap + 1) // 2
            prev = cur
        gap = N - prev
        s += gap * (gap + 1) // 2
        miss_both = s

        T = total_sub - miss[v] - miss[v + 1] + miss_both
        S2 += T

    print(S1 - S2)


if __name__ == "__main__":
    main()