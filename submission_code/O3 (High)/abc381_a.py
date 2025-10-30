def main():
    import sys

    # Read input
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()

    # Check if length is odd
    if N % 2 == 0:
        print("No")
        return

    k = (N - 1) // 2  # Number of '1's before '/' and '2's after '/'

    if S == '1' * k + '/' + '2' * k:
        print("Yes")
    else:
        print("No")


if __name__ == "__main__":
    main()