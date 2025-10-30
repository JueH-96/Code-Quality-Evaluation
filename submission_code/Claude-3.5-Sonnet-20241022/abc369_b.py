N = int(input())
A = []
S = []
for _ in range(N):
    a, s = input().split()
    A.append(int(a))
    S.append(s)

INF = float('inf')
dp = {}

def solve(i, left, right):
    if i == N:
        return 0
        
    state = (i, left, right)
    if state in dp:
        return dp[state]
        
    ans = INF
    if S[i] == 'L':
        # Move left hand to press key A[i]
        cost = abs(A[i] - left) if left != -1 else 0
        ans = min(ans, cost + solve(i+1, A[i], right))
    else:
        # Move right hand to press key A[i]
        cost = abs(A[i] - right) if right != -1 else 0
        ans = min(ans, cost + solve(i+1, left, A[i]))
        
    dp[state] = ans
    return ans

# Try all possible initial positions
min_fatigue = INF
for left_start in range(1, 101):
    for right_start in range(1, 101):
        dp.clear()
        fatigue = solve(0, left_start, right_start)
        min_fatigue = min(min_fatigue, fatigue)

print(min_fatigue)