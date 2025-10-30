import sys
import math

def ask_question(i, j):
    print(f"? {i} {j}")
    sys.stdout.flush()
    return int(input().strip())

def main():
    N, L, R = map(int, input().strip().split())
    total_sum = 0

    # Determine the range [L, R] in terms of powers of 2
    for i in range(N + 1):
        for j in range(2**(N - i)):
            l = 2**i * j
            r = 2**i * (j + 1) - 1
            if r < L or l > R:
                continue
            if l < L:
                l = L
            if r > R:
                r = R
            total_sum += ask_question(i, j)
            total_sum %= 100

    print(f"! {total_sum}")

if __name__ == "__main__":
    main()