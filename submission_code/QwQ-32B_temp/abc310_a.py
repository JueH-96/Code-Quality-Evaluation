import sys

def main():
    # Read the first line and split into N, P, Q
    n, p, q = map(int, sys.stdin.readline().split())
    # Read the second line as list of integers for D_i
    d_list = list(map(int, sys.stdin.readline().split()))
    # Find the minimum dish price
    min_d = min(d_list)
    # Calculate both options
    option_coupon = q + min_d
    # The answer is the minimum of the two options
    print(min(p, option_coupon))

if __name__ == "__main__":
    main()