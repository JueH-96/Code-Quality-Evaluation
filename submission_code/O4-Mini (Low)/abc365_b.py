def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Pair each value with its 1-based index
    paired = [(val, idx + 1) for idx, val in enumerate(A)]
    # Sort by value in descending order
    paired.sort(key=lambda x: x[0], reverse=True)
    # The second element in this sorted list is the second largest
    _, second_largest_index = paired[1]
    print(second_largest_index)

if __name__ == "__main__":
    main()