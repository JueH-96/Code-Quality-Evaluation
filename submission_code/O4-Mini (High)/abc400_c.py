import sys
import math

def main():
    data = sys.stdin.readline().strip()
    if not data:
        return
    N = int(data)
    # Count of numbers of form 2*s^2 <= N:
    cnt_twice_square = math.isqrt(N // 2)
    # Count of squares of even integers u^2 <= N:
    # u even ⇒ u = 2*k, 1 ≤ k ≤ floor(sqrt(N)/2)
    cnt_even_square = math.isqrt(N) // 2
    print(cnt_twice_square + cnt_even_square)

if __name__ == "__main__":
    main()