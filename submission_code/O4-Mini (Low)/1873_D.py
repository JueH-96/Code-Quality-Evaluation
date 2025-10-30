import sys
import threading

def main():
    input = sys.stdin.readline
    t = int(input())
    for _ in range(t):
        n, k = map(int, input().split())
        s = input().strip()
        # collect positions of black cells (1-based)
        blacks = [i+1 for i, ch in enumerate(s) if ch == 'B']
        if not blacks:
            print(0)
            continue
        ops = 0
        i = 0
        # Greedy: each time cover from leftmost uncovered black
        # with a segment of length k (shifted right if needed).
        while i < len(blacks):
            x = blacks[i]
            # choose start so that x is covered, but segment stays within [1..n]
            start = min(x, n - k + 1)
            end = start + k - 1
            ops += 1
            # skip all blacks covered by this operation
            while i < len(blacks) and blacks[i] <= end:
                i += 1
        print(ops)

if __name__ == "__main__":
    main()