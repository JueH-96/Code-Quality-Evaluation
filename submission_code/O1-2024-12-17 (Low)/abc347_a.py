def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, K = map(int, data[:2])
    A = list(map(int, data[2:]))

    # Filter multiples of K, divide by K, and store in a list
    result = [str(x // K) for x in A if x % K == 0]

    # Print in ascending order (already sorted because A was strictly increasing)
    print(' '.join(result))

# Do not forget to call the main() function
main()