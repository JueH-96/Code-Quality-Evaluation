def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    if not data:
        return
    
    # Read target string T and number of bags N.
    it = iter(data)
    T = next(it)
    N = int(next(it))
    L = len(T)
    
    # dp[j] will represent the minimum cost (yen spent) to form the prefix T[0:j]
    INF = float('inf')
    dp = [INF] * (L + 1)
    dp[0] = 0  # empty prefix costs 0
    
    # Process each bag in order (bags are numbered 1 to N)
    for _ in range(N):
        # For each bag, the first token is the count and then the strings in the bag.
        count = int(next(it))
        bag = [next(it) for _ in range(count)]
        
        # new_dp will be the dp state after considering the current bag.
        new_dp = [INF] * (L + 1)
        for j in range(L + 1):
            # Option 1: Do nothing for this bag.
            if dp[j] < new_dp[j]:
                new_dp[j] = dp[j]
            
            # If current prefix j is unreachable, skip.
            if dp[j] == INF:
                continue
            
            # Option 2: Use one string from the bag.
            for s in bag:
                ls = len(s)
                if j + ls <= L and T[j:j+ls] == s:
                    # We can extend our prefix by s, at an extra cost of 1 yen.
                    if dp[j] + 1 < new_dp[j + ls]:
                        new_dp[j + ls] = dp[j] + 1
        dp = new_dp  # Move to the next bag.
    
    # If the full string T can be formed then dp[L] is the answer; otherwise, it's unreachable.
    result = dp[L] if dp[L] != INF else -1
    sys.stdout.write(str(result))
    
if __name__ == '__main__':
    main()