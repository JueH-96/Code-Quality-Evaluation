import sys
import bisect


def read_ints(k):
    """
    Read exactly k integers from stdin, even if they span several lines.
    """
    data = []
    while len(data) < k:
        data.extend(map(int, sys.stdin.buffer.readline().split()))
    return data


def main() -> None:
    input_ = sys.stdin.buffer.readline

    # Whole rectangle size (not used in the algorithm itself)
    W, H = map(int, input_().split())

    # Strawberries
    N = int(input_())
    strawberries = [tuple(map(int, input_().split())) for _ in range(N)]

    # Vertical cuts
    A = int(input_())
    a_list = read_ints(A)

    # Horizontal cuts
    B = int(input_())
    b_list = read_ints(B)

    # Make sure cut lists are sorted (they are in the constraints, but it costs nothing)
    a_list.sort()
    b_list.sort()

    # Count strawberries in each piece
    counts = dict()                        # key -> (column_index, row_index)
    for px, py in strawberries:
        col = bisect.bisect_right(a_list, px)   # 0 … A
        row = bisect.bisect_right(b_list, py)   # 0 … B
        key = (col, row)
        counts[key] = counts.get(key, 0) + 1

    total_pieces = (A + 1) * (B + 1)

    max_cnt = max(counts.values())
    # If some piece has no strawberry, min is 0
    if len(counts) < total_pieces:
        min_cnt = 0
    else:
        min_cnt = min(counts.values())

    print(min_cnt, max_cnt)


if __name__ == "__main__":
    main()