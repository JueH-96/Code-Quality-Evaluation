import sys

def min_count_to_exceed(limit: int, values) -> int:
    """
    Return the minimum number of largest elements whose sum exceeds `limit`.
    If it is impossible even with all elements, return len(values) + 1.
    """
    acc = 0
    for i, v in enumerate(sorted(values, reverse=True), 1):
        acc += v
        if acc > limit:
            return i
    return len(values) + 1      # “Impossible” sentinel (larger than N)

def main() -> None:
    input_data = sys.stdin.buffer.read().split()
    n = int(input_data[0])
    X = int(input_data[1])
    Y = int(input_data[2])

    # A_i list starts at index 3, B_i list starts right after A part
    A = list(map(int, input_data[3: 3 + n]))
    B = list(map(int, input_data[3 + n: 3 + 2 * n]))

    # minimum dishes needed to make sweetness exceed X
    need_sweet = min_count_to_exceed(X, A)
    # minimum dishes needed to make saltiness exceed Y
    need_salt  = min_count_to_exceed(Y, B)

    # If neither dimension can exceed, he will simply eat all dishes.
    ans = min(need_sweet, need_salt, n)
    print(ans)

if __name__ == "__main__":
    main()