def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, K = map(int, data[:2])
    A = list(map(int, data[2:]))

    # Calculate the new order:
    # Take K cards from the bottom, place them on top
    result = A[-K:] + A[:-K]

    # Print result
    print(*result)

# Do not forget to call main
main()