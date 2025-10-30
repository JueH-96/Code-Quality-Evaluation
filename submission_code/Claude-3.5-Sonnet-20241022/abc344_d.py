def solve():
    T = input()
    N = int(input())
    bags = []
    for _ in range(N):
        line = input().split()
        A = int(line[0])
        strings = line[1:]
        bags.append(strings)
    
    # dp[i][j] = minimum cost to make prefix of length j after using first i bags
    dp = {}
    
    def get_dp(i, j):
        if (i,j) not in dp:
            return float('inf')
        return dp[(i,j)]
    
    # initialize base case
    dp[(0,0)] = 0
    
    # for each bag
    for i in range(N):
        # for each prefix length possible
        for j in range(len(T)+1):
            # try not using current bag
            dp[(i+1,j)] = get_dp(i,j)
            
            # try using each string from current bag
            for s in bags[i]:
                if j >= len(s) and T[j-len(s):j] == s:
                    dp[(i+1,j)] = min(dp[(i+1,j)], get_dp(i,j-len(s)) + 1)
    
    ans = get_dp(N, len(T))
    if ans == float('inf'):
        print(-1)
    else:
        print(ans)

solve()