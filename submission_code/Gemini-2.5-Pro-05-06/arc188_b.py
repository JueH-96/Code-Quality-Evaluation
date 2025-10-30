import math

def solve():
    N, K = map(int, input().split())

    if 2 * K == N: # Case 1: Alice and Bob are diametrically opposite
        if N == 2: # N=2, K=1
            print("Yes")
        else: # N > 2
            print("No")
    else: # Case 2: Alice and Bob are not diametrically opposite
        if N % 2 == 1: # Subcase 2a: N is odd
            if math.gcd(N, K) == 1:
                print("Yes")
            else:
                print("No")
        else: # Subcase 2b: N is even
            # This implies N % 4 == 0 or N % 4 == 2
            if N % 4 == 2: # N = 2 * (odd number)
                # N/2 is odd. We need gcd(N/2, K) == 1 for all points to be colored.
                if math.gcd(N // 2, K) == 1:
                    print("Yes")
                else:
                    print("No")
            else: # N % 4 == 0. N = 4 * (some number)
                  # N/2 is even. This configuration leads to "No".
                print("No")

T = int(input())
for _ in range(T):
    solve()