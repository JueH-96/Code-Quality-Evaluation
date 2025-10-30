import sys

def main() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    n = int(data[0])
    nums = list(map(int, data[1:1 + n]))

    # Pair each value with its 1-based position.
    indexed = [(value, idx + 1) for idx, value in enumerate(nums)]
    # Sort by value descending.
    indexed.sort(reverse=True, key=lambda x: x[0])

    # Second largest element's original position.
    second_largest_pos = indexed[1][1]
    print(second_largest_pos)

if __name__ == "__main__":
    main()