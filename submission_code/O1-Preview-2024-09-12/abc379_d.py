# YOUR CODE HERE

import sys
import bisect

def main():
    import sys
    import threading

    def run():
        Q = int(sys.stdin.readline())
        time = 0  # Current time
        plant_times = []  # List of plant times, sorted

        outputs = []

        for _ in range(Q):
            tokens = sys.stdin.readline().split()
            if tokens[0] == '1':
                # Type 1 query: Plant a new plant at current time
                bisect.insort_right(plant_times, time)
            elif tokens[0] == '2':
                # Type 2 query: Wait for T days
                T = int(tokens[1])
                time += T
            elif tokens[0] == '3':
                # Type 3 query: Harvest plants with height >= H
                H = int(tokens[1])
                threshold_time = time - H
                # Find index of first plant_time > threshold_time
                idx = bisect.bisect_right(plant_times, threshold_time)
                cnt = idx
                # Remove harvested plants
                del plant_times[0:idx]
                # Record the output
                outputs.append(str(cnt))
        for out in outputs:
            print(out)

    threading.Thread(target=run).start()

main()