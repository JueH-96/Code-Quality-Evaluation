import math

T = int(input())
for _ in range(T):
    N, K = map(int, input().split())
    g = math.gcd(N, 2 * K)
    if (N // g) % 2 == 1:
        print("Yes")
    else:
        print("No")