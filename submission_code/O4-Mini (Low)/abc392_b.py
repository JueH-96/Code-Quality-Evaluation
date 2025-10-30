def main():
    import sys
    data = sys.stdin.read().split()
    N, M = map(int, data[:2])
    A = set(map(int, data[2:]))

    # Find all integers from 1 to N that are not in A
    missing = [str(x) for x in range(1, N + 1) if x not in A]
    count = len(missing)

    # Output the results
    print(count)
    if count > 0:
        print(" ".join(missing))

if __name__ == "__main__":
    main()