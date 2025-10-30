def main():
    import sys

    # Read input
    N = int(sys.stdin.readline().strip())

    # Compute the nearest water station at multiples of 5 km from 0 to 100.
    # Let a be the largest multiple of 5 not exceeding N, and b = a + 5 (if within 100).
    a = (N // 5) * 5
    b = a + 5 if a + 5 <= 100 else a

    # Compare distances to pick the nearest
    if abs(N - a) <= abs(N - b):
        print(a)
    else:
        print(b)


if __name__ == "__main__":
    main()