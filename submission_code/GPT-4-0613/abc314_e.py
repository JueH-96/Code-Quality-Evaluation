import heapq
import sys

def main():
    N, M = map(int, input().split())
    wheels = []
    for _ in range(N):
        C, P, *S = map(int, input().split())
        S.sort(reverse=True)
        S.append(0)
        acc = [0]*(P+1)
        for i in range(P):
            acc[i+1] = acc[i] + S[i]
        wheels.append((C, P, S, acc))
    dp = [0]*(M+1)
    for i in range(1, M+1):
        dp[i] = min(C*(i*P - acc[min(i, P)]) for C, P, S, acc in wheels)
        for j in range(1, i):
            dp[i] = min(dp[i], dp[j] + min(C*((i-j)*P - acc[min(i-j, P)]) for C, P, S, acc in wheels))
    print(dp[M])

if __name__ == "__main__":
    main()