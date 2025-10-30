import sys
import threading
def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    MAX = 10**6
    cnt = [0] * (MAX + 1)
    for x in A:
        cnt[x] += 1

    # Build prefix sums for cnt
    pre = [0] * (MAX + 1)
    s = 0
    for i in range(MAX + 1):
        s += cnt[i]
        pre[i] = s

    ans = 0
    # For each possible minimum value m
    for m in range(1, MAX + 1):
        c_m = cnt[m]
        if c_m == 0:
            continue

        # Pairs where both elements == m contribute floor(m/m)=1
        if c_m > 1:
            ans += c_m * (c_m - 1) // 2  # each such pair adds 1

        # Pairs where min = m, max = b > m
        # We group b by k = floor(b/m)
        # For k from 1 up to MAX//m
        # note k=1 includes b in [m,2m-1], so subtract the m==b case
        limit = MAX // m
        for k in range(1, limit + 1):
            l = k * m
            r = min((k + 1) * m - 1, MAX)
            cnt_b = pre[r] - (pre[l - 1] if l > 0 else 0)
            if k == 1:
                # subtract the cases b == m counted in cnt_b
                cnt_b -= c_m
            if cnt_b > 0:
                ans += c_m * cnt_b * k

    print(ans)

if __name__ == "__main__":
    main()