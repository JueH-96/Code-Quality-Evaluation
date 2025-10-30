import sys
import threading

def main():
    import sys

    data = sys.stdin
    n, K = map(int, data.readline().split())
    meds = []
    total_b = 0
    for _ in range(n):
        a, b = map(int, data.readline().split())
        meds.append((a, b))
        total_b += b

    # If already <= K on day 1
    if total_b <= K:
        print(1)
        return

    # We need to find the earliest day d > 1 such that the sum of pills
    # on day d is <= K. On day d, only medicines with a_i >= d are taken.
    # Equivalently, let R = total_b - K. We need to drop at least R pills
    # from the total. Dropping pills means skipping all medicines with a_i < d.
    # Let prefix_sum(x) = sum of b_i for medicines with a_i <= x.
    # We want the smallest x such that prefix_sum(x) >= R, then d = x + 1.
    R = total_b - K

    # Sort medicines by their end day a_i
    meds.sort(key=lambda x: x[0])

    prefix = 0
    for a, b in meds:
        prefix += b
        if prefix >= R:
            # Once we've dropped enough pills by day a, the first
            # day with <= K pills is a+1
            print(a + 1)
            return

if __name__ == "__main__":
    main()