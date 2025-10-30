def solve():
    N = int(input())
    A = list(map(int, input().split()))
    ST = []
    for _ in range(N-1):
        s, t = map(int, input().split())
        ST.append((s, t))
    
    # For each currency i, try all possible conversions to get max amount of currency i+1
    for i in range(N-1):
        s, t = ST[i]
        # Convert as many times as possible from currency i to i+1
        conversions = A[i] // s
        A[i+1] += conversions * t
    
    return A[N-1]

print(solve())