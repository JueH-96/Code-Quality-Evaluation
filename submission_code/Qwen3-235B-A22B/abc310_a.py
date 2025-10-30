# Read input
import sys

def main():
    # Read first line
    N, P, Q = map(int, sys.stdin.readline().split())
    # Read second line
    D = list(map(int, sys.stdin.readline().split()))
    # Find minimum dish price
    min_d = min(D)
    # Calculate coupon option
    coupon_total = Q + min_d
    # Determine and print the minimum cost
    print(min(P, coupon_total))

if __name__ == "__main__":
    main()