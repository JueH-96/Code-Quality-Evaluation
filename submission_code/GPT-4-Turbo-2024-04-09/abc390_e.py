def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    X = int(data[1])
    
    vitamins = [[], [], []]
    
    index = 2
    for _ in range(N):
        V = int(data[index]) - 1
        A = int(data[index + 1])
        C = int(data[index + 2])
        vitamins[V].append((A, C))
        index += 3
    
    # Sort each vitamin list by cost (ascending)
    for v in range(3):
        vitamins[v].sort(key=lambda x: x[1])
    
    # Dynamic programming approach to find the maximum minimum vitamin intake
    # dp[c] will store the maximum amount of each vitamin we can get with exactly c calories
    dp = [[-float('inf')] * 3 for _ in range(X + 1)]
    dp[0] = [0, 0, 0]  # 0 calories, 0 units of each vitamin
    
    for v in range(3):
        for amount, cost in vitamins[v]:
            for c in range(X, cost - 1, -1):
                dp[c][v] = max(dp[c][v], dp[c - cost][v] + amount)
    
    # We need to find the maximum value of min(dp[c][0], dp[c][1], dp[c][2]) for all c <= X
    max_min_vitamin = 0
    for c in range(X + 1):
        if min(dp[c]) > max_min_vitamin:
            max_min_vitamin = min(dp[c])
    
    print(max_min_vitamin)