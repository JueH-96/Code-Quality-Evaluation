import sys
input = sys.stdin.readline

def main():
    N, K = map(int, input().split())
    A = list(map(int, input().split()))
    B = [0] + A + [N+1]
    diff = [B[i+1] - B[i] - 1 for i in range(K+1)]
    diff.sort()
    ans = 0
    if 2*N-K % 2 == 0:
        for i in range(K+1-(2*N-K)//2, (2*N-K)//2+1):
            ans += diff[i]
    else:
        for i in range(K+1-(2*N-K)//2, (2*N-K)//2+2):
            ans += diff[i]
    print(ans)
    
main()