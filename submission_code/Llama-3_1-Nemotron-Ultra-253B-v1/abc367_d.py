def main():
    import sys
    from collections import defaultdict

    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    prefix = [0] * (N + 1)
    for i in range(N):
        prefix[i + 1] = prefix[i] + A[i]
    total = prefix[-1]
    g = gcd(total, M)
    cnt = defaultdict(int)
    cnt[0] = 1  # S_0 = 0
    ans = 0
    for i in range(1, N + 1):
        remainder = prefix[i - 1] % g
        ans += cnt[remainder]
        cnt[remainder] += 1
    print(ans)

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

main()