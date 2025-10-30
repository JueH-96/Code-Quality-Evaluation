def main():
    import sys
    input = sys.stdin.readline
    T = input().strip()
    N = int(input().strip())
    
    # Initialize DP where dp[j] = minimum cost to form prefix T[:j]
    dp = [float('inf')] * (len(T) + 1)
    dp[0] = 0  # cost of forming empty prefix is 0
    
    # Process each bag in order
    for _ in range(N):
        data = input().split()
        A = int(data[0])
        strings = data[1:]
        # We'll compute new dp after processing current bag
        next_dp = [float('inf')] * (len(T) + 1)
        for j in range(len(T) + 1):
            # Option 2: Do nothing in this bag bag
            if dp[j] < next_dp[j]:
                next_dp[j] = dp[j]
            # Option 1: Pick one string from current bag if possible
            if dp[j] != float('inf'):
                for s in strings:
                    # Check if s can match T starting at position j
                    if j + len(s) <= len(T) and T[j:j+len(s)] == s:
                        if dp[j] + 1 < next_dp[j + len(s)]:
                            next_dp[j + len(s)] = dp[j] + 1
        dp = next_dp

    # The answer is dp[len(T)] if it is not "infinity"
    print(dp[len(T)] if dp[len(T)] != float('inf') else -1)

if __name__ == '__main__':
    main()