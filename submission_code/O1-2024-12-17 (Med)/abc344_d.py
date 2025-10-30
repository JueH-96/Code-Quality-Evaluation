def main():
    import sys

    data = sys.stdin.read().strip().split()
    T = data[0]
    N = int(data[1])
    index = 2
    
    bags = []
    for _ in range(N):
        A = int(data[index])
        index += 1
        bag_strings = data[index:index+A]
        index += A
        bags.append(bag_strings)
    
    lenT = len(T)
    INF = float('inf')
    
    # dp[i][j] represents the minimum cost to form T[:j] after processing bags up to i (0-based).
    # i ranges from 0 to N (inclusive) meaning i is the number of bags processed.
    # j ranges from 0 to lenT (inclusive) meaning j is the length of the prefix of T formed.
    dp = [[INF] * (lenT + 1) for _ in range(N + 1)]
    dp[0][0] = 0  # 0 cost to form empty string with 0 bags processed
    
    # Fill the DP table
    for i in range(N):
        for j in range(lenT + 1):
            if dp[i][j] == INF:
                continue
            
            # Option 1: Do nothing for bag i
            dp[i+1][j] = min(dp[i+1][j], dp[i][j])
            
            # Option 2: Pay 1 yen to use one string from bag i
            for s in bags[i]:
                if j + len(s) <= lenT and T[j:j+len(s)] == s:
                    dp[i+1][j + len(s)] = min(dp[i+1][j + len(s)], dp[i][j] + 1)
    
    # The answer is the minimum cost to form the entire string T after all N bags
    answer = dp[N][lenT]
    print(answer if answer != INF else -1)

# Do not forget to call main
if __name__ == "__main__":
    main()