import sys

def main():
    input = sys.stdin.readline
    T = input().strip()
    N_line = input().strip()
    # If N line is empty (in case of trailing newlines), read again
    while N_line == "":
        N_line = input().strip()
    N = int(N_line)
    bags = []
    for _ in range(N):
        parts = input().strip().split()
        # parts[0] is A_i, then that many strings
        # We don't actually need A_i except to parse
        strs = parts[1:]
        bags.append(strs)

    L = len(T)
    INF = 10**9
    # dp[pos] = minimum cost to have matched pos chars of T
    dp = [INF] * (L + 1)
    dp[0] = 0

    for bag in bags:
        # copy old dp for skip transitions
        new_dp = dp[:]  # skip taking any string from this bag
        # try taking one string from this bag
        for pos in range(L + 1):
            cost = dp[pos]
            if cost == INF:
                continue
            # from pos, try each string
            # check if T[pos:pos+len(s)] == s
            for s in bag:
                slen = len(s)
                end = pos + slen
                if end <= L and T[pos:end] == s:
                    # can match
                    if new_dp[end] > cost + 1:
                        new_dp[end] = cost + 1
        dp = new_dp

    ans = dp[L]
    if ans >= INF:
        print(-1)
    else:
        print(ans)

if __name__ == "__main__":
    main()