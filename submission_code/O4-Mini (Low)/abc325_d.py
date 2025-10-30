import sys
import threading

def main():
    data = sys.stdin
    N = int(data.readline().strip())
    intervals = []
    for _ in range(N):
        t, d = map(int, data.readline().split())
        # product is in range [t, t+d], inclusive
        intervals.append((t, t + d))
    # sort by end time (r)
    intervals.sort(key=lambda x: x[1])
    # cur is last print time
    # we need next print time >= cur + 1
    cur = -10**30
    count = 0
    for l, r in intervals:
        # earliest we can print this product
        t = max(l, cur + 1)
        if t <= r:
            count += 1
            cur = t
    print(count)

if __name__ == "__main__":
    main()