def main():
    import sys

    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:]))

    # Pair each element with its 1-based index
    pairs = [(val, idx + 1) for idx, val in enumerate(A)]
    # Sort by value descending
    pairs.sort(key=lambda x: x[0], reverse=True)
    # The second element in this sorted list is the second largest
    print(pairs[1][1])


if __name__ == "__main__":
    main()