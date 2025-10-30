import sys
input = sys.stdin.readline

def main():
    N = int(input())
    A = list(map(int, input().split()))

    ans = 0
    max_a = 0
    next_max_sum = 0
    max_sum = 0
    for i in range(N):
        max_a = max(max_a, A[N-i-1])
        next_max_sum += max_a - A[N-i-1]
    for i in range(N-1):
        ans += next_max_sum
        next_max_sum -= max_sum - A[i]
        max_sum += A[i]
    print(ans)