import sys
import math

def solve():
    input = sys.stdin.read().split()
    T = int(input[0])
    idx = 1
    for _ in range(T):
        N = int(input[idx])
        K = int(input[idx+1])
        idx += 2
        
        if N % 2 == 0 and K == N // 2:
            print("No")
            continue
        
        g = math.gcd(N, 2 * K)
        m = N // g
        if m % 2 == 1:
            print("Yes")
        else:
            print("No")

if __name__ == '__main__':
    solve()