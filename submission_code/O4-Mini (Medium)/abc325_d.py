import sys
import threading

def main():
    import sys
    data = sys.stdin
    n_line = data.readline().strip()
    if not n_line:
        return
    N = int(n_line)
    intervals = []
    # Read products, form intervals [start, end]
    for _ in range(N):
        line = data.readline().split()
        t = int(line[0])
        d = int(line[1])
        # product is in printer range from t to t+d
        intervals.append((t, t + d))
    # Sort by end time ascending, and for ties by start ascending
    intervals.sort(key=lambda x: (x[1], x[0]))
    last_time = -10**30  # last print time
    count = 0
    # Greedily schedule prints
    for s, e in intervals:
        # earliest we can print this one
        t = s if s > last_time + 1 else last_time + 1
        # check if it fits before the product leaves
        if t <= e:
            count += 1
            last_time = t
    # Output the maximum count
    print(count)

if __name__ == "__main__":
    main()