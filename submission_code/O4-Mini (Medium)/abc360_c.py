import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    W = [int(next(it)) for _ in range(n)]
    # total sum of weights
    total = 0
    # track max weight in each box
    max_in_box = [0] * (n + 1)
    # track if box has any items
    has = [False] * (n + 1)
    for ai, wi in zip(A, W):
        total += wi
        if not has[ai]:
            has[ai] = True
            max_in_box[ai] = wi
        else:
            if wi > max_in_box[ai]:
                max_in_box[ai] = wi
    sum_max = 0
    # sum up the kept items (one per non-empty box)
    for box in range(1, n + 1):
        if has[box]:
            sum_max += max_in_box[box]
    # answer is total weights minus weights we keep
    print(total - sum_max)

if __name__ == "__main__":
    main()