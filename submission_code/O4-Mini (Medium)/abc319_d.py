import sys
import threading

def main():
    import sys

    data = sys.stdin.read().split()
    N = int(data[0])
    M = int(data[1])
    L = list(map(int, data[2:]))

    # If we can put each word on its own line (M >= N), the answer is just the longest word.
    max_len = max(L)
    if M >= N:
        print(max_len)
        return

    # Binary search for the minimum width W
    # Lower bound is the max word length, upper bound is sum of lengths + spaces
    lo = max_len
    hi = sum(L) + (N - 1)

    def lines_needed(W):
        # Greedy packing: count how many lines needed with width W
        lines = 1
        current_width = 0
        for length in L:
            if current_width == 0:
                # start of a line
                current_width = length
            else:
                # need a space before this word
                if current_width + 1 + length <= W:
                    current_width += 1 + length
                else:
                    lines += 1
                    current_width = length
        return lines

    while lo < hi:
        mid = (lo + hi) // 2
        if lines_needed(mid) <= M:
            hi = mid
        else:
            lo = mid + 1

    print(lo)

if __name__ == "__main__":
    main()