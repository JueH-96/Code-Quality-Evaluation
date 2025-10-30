def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    T = input_data[0]
    N = int(input_data[1])
    idx = 2
    bags = []
    for _ in range(N):
        A = int(input_data[idx])
        idx += 1
        bag = input_data[idx: idx + A]
        idx += A
        bags.append(bag)

    m = len(T)
    INF = 10**9
    # dp[j] = minimum cost to obtain prefix T[:j] using processed bags so far.
    dp = [INF] * (m + 1)
    dp[0] = 0

    for bag in bags:
        new_dp = [INF] * (m + 1)
        for j in range(m + 1):
            if dp[j] == INF:
                continue
            # Option 1: Do nothing with the current bag.
            if dp[j] < new_dp[j]:
                new_dp[j] = dp[j]
            # Option 2: Pick one string from current bag.
            for s in bag:
                ls = len(s)
                if j + ls <= m and T[j:j + ls] == s:
                    # Valid extension; update cost by 1 yen.
                    if dp[j] + 1 < new_dp[j + ls]:
                        new_dp[j + ls] = dp[j] + 1
        dp = new_dp

    ans = dp[m]
    if ans == INF:
        sys.stdout.write("-1")
    else:
        sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()