K = int(input())
S = input().strip()
T = input().strip()

m, n = len(S), len(T)

# If the difference in lengths is greater than K, it's impossible
if abs(m - n) > K:
    print("No")
else:
    # Space-optimized DP
    prev = list(range(n + 1))
    
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            if S[i-1] == T[j-1]:
                curr[j] = prev[j-1]
            else:
                curr[j] = 1 + min(prev[j], curr[j-1], prev[j-1])
        prev = curr
    
    if prev[n] <= K:
        print("Yes")
    else:
        print("No")