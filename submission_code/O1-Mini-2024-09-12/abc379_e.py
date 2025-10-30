def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1]
    
    total = 0
    dp = 0
    for i in range(1, N+1):
        digit = int(S[i-1])
        dp = dp * 10 + digit * i
        total += dp
    print(total)