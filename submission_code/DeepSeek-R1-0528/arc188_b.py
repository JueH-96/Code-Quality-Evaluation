import math

T = int(input().strip())
for _ in range(T):
    N, K = map(int, input().split())
    d = math.gcd(2 * K, N)
    G = math.gcd(2 * d, N)
    g2 = math.gcd(2, G)
    G1 = G // g2
    
    if G1 == 1:
        print("Yes")
    elif G1 == 2:
        if K % 2 == 1:
            print("Yes")
        else:
            print("No")
    else:
        print("No")