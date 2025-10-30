import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    k = int(data[1])
    A = list(map(int, data[2:2+k]))
    
    lost = [False] * (n + 1)
    for a in A:
        lost[a] = True
        
    L = []
    for i in range(1, n + 1):
        if lost[i]:
            L.append(i)
        else:
            L.append(i)
            L.append(i)
            
    M = len(L)
    if M == 0:
        print(0)
        return
        
    dp = [0] * (M + 1)
    dp[0] = 0
    if M >= 1:
        dp[1] = 0
        
    for i in range(2, M + 1):
        if i % 2 == 0:
            dp[i] = dp[i - 2] + (L[i - 1] - L[i - 2])
        else:
            option1 = dp[i - 1]
            option2 = dp[i - 2] + (L[i - 1] - L[i - 2])
            dp[i] = min(option1, option2)
            
    print(dp[M])

if __name__ == '__main__':
    main()