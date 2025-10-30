import math

def solve(N, L, R):
    m = 0
    total = 0

    for i in range(N+1):
        for j in range(2**(N-i)):
            l = 2**i * j
            r = 2**i * (j+1) - 1
            if l <= R and r >= L:
                m += 1
                print(f"? {i} {j}")
                remainder = int(input())
                if remainder == -1:
                    return
                total += remainder
                if l >= L and r <= R:
                    total -= remainder
                    total %= 100

    print(f"! {total % 100}")

N, L, R = map(int, input().split())
solve(N, L, R)