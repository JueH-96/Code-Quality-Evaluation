import sys, bisect

def main() -> None:
    data = sys.stdin.buffer.read().split()
    N = int(data[0])
    T = data[1]                # bytes object
    M = len(T)

    A = [0] * N                # longest matched prefix length of T for each S_i
    B = [0] * N                # longest matched  suffix length of T for each S_i

    idx = 2                    # current position while traversing `data`
    for i in range(N):
        s = data[idx]
        idx += 1

        # longest prefix of T that is a subsequence of s
        p = 0
        for ch in s:
            if p < M and ch == T[p]:
                p += 1
                if p == M:
                    break
        A[i] = p

        # longest suffix of T that is a subsequence of s
        q = M - 1
        for ch in reversed(s):
            if q >= 0 and ch == T[q]:
                q -= 1
                if q < 0:
                    break
        B[i] = M - 1 - q       # matched characters from the tail of T

    # sort all suffix-capabilities for binary search
    B_sorted = sorted(B)

    ans = 0
    for a in A:
        need = M - a           # minimal suffix length still necessary
        if need <= 0:          # S_i alone already contains T
            ans += N
        else:
            pos = bisect.bisect_left(B_sorted, need)
            ans += N - pos     # B_j large enough

    print(ans)


if __name__ == "__main__":
    main()