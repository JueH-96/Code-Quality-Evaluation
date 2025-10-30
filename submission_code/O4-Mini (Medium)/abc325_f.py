import sys
import threading
def main():
    import sys
    input = sys.stdin.readline
    N = int(input().strip())
    D = list(map(int, input().split()))
    L1, C1, K1 = map(int, input().split())
    L2, C2, K2 = map(int, input().split())
    # We will DP on type1-count dimension: make K1 the smaller bound
    # Swap types if needed
    if K1 > K2:
        L1, L2 = L2, L1
        C1, C2 = C2, C1
        K1, K2 = K2, K1
    # Precompute for each section the useful (k, g) pairs:
    # k = number of type1 sensors, g = type2 sensors needed for that section
    from math import ceil
    INF = 10**18
    sect = []
    for Di in D:
        # max type1 for this section
        m1 = (Di + L1 - 1) // L1
        if m1 > K1:
            m1 = K1
        pairs = []
        last_g = None
        # compute g for k=0..m1, but only when it strictly decreases
        for k in range(0, m1+1):
            rem = Di - k * L1
            if rem <= 0:
                g = 0
            else:
                g = (rem + L2 - 1) // L2
            # cap at K2+1 for pruning
            if g > K2+1:
                g = K2+1
            # record if first or decreased
            if last_g is None or g < last_g:
                pairs.append((k, g))
                last_g = g
                if g == 0:
                    break
        sect.append(pairs)
    # Group DP: dp_prev[x1] = minimal total type2 sensors needed after processing some sections with x1 type1 sensors
    dp_prev = [INF] * (K1 + 1)
    dp_prev[0] = 0
    # iterate sections
    for pairs in sect:
        # new dp
        dp_cur = [INF] * (K1 + 1)
        # for each choice (k_i, g_i) in this section
        # we do dp_cur[x1] = min(dp_cur[x1], dp_prev[x1 - k_i] + g_i)
        # for all x1 >= k_i
        for k_i, g_i in pairs:
            # local refs
            add = g_i
            # for a1 from k_i..K1
            # we take prev = dp_prev[a1 - k_i] + add
            # compare to dp_cur[a1]
            # inline for speed
            prev_row = dp_prev
            cur_row = dp_cur
            # range iteration
            start = k_i
            # manually unroll?
            for a1 in range(start, K1 + 1):
                v = prev_row[a1 - k_i] + add
                if v < cur_row[a1]:
                    cur_row[a1] = v
        dp_prev = dp_cur
    # Now dp_prev[x1] is minimal type2 sensors for each total x1 of type1 sensors
    ans = INF
    for x1 in range(K1 + 1):
        y2 = dp_prev[x1]
        if y2 <= K2:
            cost = x1 * C1 + y2 * C2
            if cost < ans:
                ans = cost
    if ans >= INF:
        print(-1)
    else:
        print(ans)

if __name__ == "__main__":
    main()