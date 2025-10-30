import sys

def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    if not data:
        return
    N, M = data[0], data[1]
    L = data[2:]
    
    def lines_needed(width: int) -> int:
        """Return how many lines are required if the window width is `width`."""
        lines = 1
        cur = 0  # current line width
        for w in L:
            if cur == 0:                # first word in the line
                cur = w
            elif cur + 1 + w <= width:  # can put it in the same line
                cur += 1 + w
            else:                       # move to a new line
                lines += 1
                cur = w
        return lines

    low = max(L)                       # at least the longest single word
    high = sum(L) + (N - 1)            # width to place everything in one line

    while low < high:
        mid = (low + high) // 2
        if lines_needed(mid) <= M:
            high = mid                 # mid is feasible, try to shrink
        else:
            low = mid + 1              # mid is too small
    print(low)

if __name__ == "__main__":
    main()