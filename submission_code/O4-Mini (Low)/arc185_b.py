def main():
    import sys
    input = sys.stdin.readline

    T = int(input())
    out = []
    for _ in range(T):
        n = int(input())
        A = list(map(int, input().split()))
        total = sum(A)
        # the largest possible value for any A[i] in a non-decreasing
        # rearrangement reachable by only moving "mass" left is
        # floor(total / n).  We must have for all k:
        #   sum(A[0..k-1]) <= k * floor(total/n)
        # otherwise we cannot boost the prefix enough.
        limit = total // n
        ps = 0
        ok = True
        for i, a in enumerate(A, start=1):
            ps += a
            if ps > i * limit:
                ok = False
                break
        out.append("Yes" if ok else "No")

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()