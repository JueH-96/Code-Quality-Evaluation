N = int(input())
S = input().strip()
C = list(map(int, input().split()))

min_cost = float('inf')

for k in range(N - 1):
    for x in [0, 1]:
        cost = 0
        
        # S[k] = S[k+1] = x
        if int(S[k]) != x:
            cost += C[k]
        if int(S[k + 1]) != x:
            cost += C[k + 1]
        
        # Pattern from position 0 to k-1: alternating ending with 1-x
        for i in range(k):
            target = (1 - x) if (k - 1 - i) % 2 == 0 else x
            if int(S[i]) != target:
                cost += C[i]
        
        # Pattern from position k+2 to N-1: alternating starting with 1-x
        for i in range(k + 2, N):
            target = (1 - x) if (i - k - 2) % 2 == 0 else x
            if int(S[i]) != target:
                cost += C[i]
        
        min_cost = min(min_cost, cost)

print(min_cost)