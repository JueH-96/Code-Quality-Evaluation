import sys, heapq

def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    q = data[0]
    idx = 1

    # min-heap that stores the time offset when each plant was planted
    pots = []           # heap of insertion offsets
    current_time = 0    # total number of days that have elapsed
    answers = []

    for _ in range(q):
        typ = data[idx]
        idx += 1

        if typ == 1:                       # add a new plant
            heapq.heappush(pots, current_time)

        elif typ == 2:                     # time passes
            t = data[idx]
            idx += 1
            current_time += t

        else:                              # harvest
            h = data[idx]
            idx += 1
            threshold = current_time - h   # plants with insertion_time ≤ threshold are tall enough
            harvested = 0
            while pots and pots[0] <= threshold:
                heapq.heappop(pots)
                harvested += 1
            answers.append(str(harvested))

    sys.stdout.write('
'.join(answers))

if __name__ == "__main__":
    main()