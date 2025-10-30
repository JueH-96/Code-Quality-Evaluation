import sys
input = sys.stdin.readline

N = int(input())
A = list(map(int, input().split()))

for i in range(N-1):
    ans = A[i]*A[i+1]
    print(ans, end=" ")