import sys

def main():
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    
    if K == 0:
        print(0)
        return
    
    prev_prev = 0  # dp[0]
    prev = 0       # dp[1]
    
    for i in range(2, K + 1):
        if i % 2 == 0:
            current = prev_prev + (A[i-1] - A[i-2])
        else:
            current = min(prev, prev_prev + (A[i-1] - A[i-2]))
        prev_prev, prev = prev, current
    
    print(prev)

if __name__ == "__main__":
    main()