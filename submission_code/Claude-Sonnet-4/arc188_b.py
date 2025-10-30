import math

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

T = int(input())
for _ in range(T):
    N, K = map(int, input().split())
    g = gcd(N, K)
    if (N // g) % 2 == 1:
        print("Yes")
    else:
        print("No")