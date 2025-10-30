import sys
import threading

def main():
    import sys

    T = sys.stdin.readline().strip()
    N_line = sys.stdin.readline().strip()
    # If there's no N line, input is malformed, but per problem it won't happen.
    N = int(N_line)
    bags = []
    for _ in range(N):
        parts = sys.stdin.readline().split()
        A = int(parts[0])
        strs = parts[1:]
        bags.append(strs)

    L = len(T)
    INF = 10**9
    # dp[i][k]: after first i bags, minimal cost to match first k chars of T
    # We only need two rows rolling.
    prev = [INF] * (L + 1)
    prev[0] = 0

    for i in range(1, N+1):
        curr = [INF] * (L + 1)
        # Option: do nothing with bag i
        for k in range(L+1):
            if prev[k] < curr[k]:
                curr[k] = prev[k]
        # Option: pick one string from bag i
        for k in range(L+1):
            if prev[k] == INF:
                continue
            base_cost = prev[k] + 1
            # try each string in bag i-1 (0-indexed)
            for s in bags[i-1]:
                slen = len(s)
                if k + slen <= L and T[k:k+slen] == s:
                    if base_cost < curr[k + slen]:
                        curr[k + slen] = base_cost
        prev = curr

    ans = prev[L]
    if ans >= INF:
        print(-1)
    else:
        print(ans)

if __name__ == "__main__":
    main()