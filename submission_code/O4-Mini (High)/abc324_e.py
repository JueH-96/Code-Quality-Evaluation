import sys

def main():
    input = sys.stdin.readline
    first = input().split()
    if not first:
        return
    N = int(first[0])
    T = first[1].strip()
    m = len(T)
    # Precompute reversed T for suffix matching
    T_rev = T[::-1]

    # count_g[x] = number of strings S_j whose maximal matched suffix length is x
    count_g = [0] * (m + 1)
    # f_list[i] = maximal prefix length of T matched in S_i
    f_list = [0] * N

    for i in range(N):
        s = input().strip()
        # Compute f_i: how many characters of T (from start) can be matched in s
        p = 0
        for c in s:
            if p < m and c == T[p]:
                p += 1
        f_list[i] = p

        # Compute g_i: how many characters of T (from end) can be matched in s
        q = 0
        for c in reversed(s):
            if q < m and c == T_rev[q]:
                q += 1
        count_g[q] += 1

    # Build suffix sums of count_g to answer "how many g_j >= r?"
    # suff[r] = sum_{x >= r} count_g[x]
    suff = [0] * (m + 1)
    suff[m] = count_g[m]
    for r in range(m - 1, -1, -1):
        suff[r] = suff[r + 1] + count_g[r]

    # For each f_i, we need g_j >= m - f_i
    ans = 0
    for f in f_list:
        need = m - f
        # need is in [0..m], so suff[need] is valid
        ans += suff[need]

    print(ans)

if __name__ == "__main__":
    main()