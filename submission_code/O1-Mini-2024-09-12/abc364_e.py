import sys
import sys
def main():
    import sys
    import sys
    N, X, Y = map(int, sys.stdin.readline().split())
    dishes = [tuple(map(int, sys.stdin.readline().split())) for _ in range(N)]
    
    INF = Y + 1
    dp = [ [INF]*(X+1) for _ in range(N+1)]
    dp[0][0] = 0
    
    for A, B in dishes:
        for k in range(N, 0, -1):
            for a in range(X - A +1):
                if dp[k-1][a] + B < dp[k][a + A]:
                    dp[k][a + A] = dp[k-1][a] + B
                    
    for k in range(N, 0, -1):
        for a in range(X+1):
            if dp[k][a] <= Y:
                print(k)
                return
    print(0)

if __name__ == "__main__":
    main()