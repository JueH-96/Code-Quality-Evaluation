def inv(x):
    return pow(x, 998244353-2, 998244353)

def main():
    N = int(input())
    A = list(map(int, input().split()))
    divisor = N
    ans = A[0]
    for i in range(1, N):
        ans += (A[i]-A[i-1]) * (i+1) % divisor * A[i]
        divisor = inv(divisor)
    print(ans % 998244353)

main()