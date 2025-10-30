# YOUR CODE HERE
import sys
import math

def main():
    import sys
    import threading
    def solve():
        N_str, M_str, K_str = sys.stdin.read().split()
        N = int(N_str)
        M = int(M_str)
        K = int(K_str)

        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a

        def lcm(a, b):
            return a * b // gcd(a, b)

        N, M = int(N), int(M)
        L = lcm(N, M)

        def f(x):
            return x // N + x // M - 2 * (x // L)

        low, high = 1, K * max(N, M)

        while low < high:
            mid = (low + high) // 2
            if f(mid) >= K:
                high = mid
            else:
                low = mid + 1
        print(low)

    threading.Thread(target=solve).start()

if __name__ == "__main__":
    main()