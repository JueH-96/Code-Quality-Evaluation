import sys
import bisect

def main():
    input = sys.stdin.read
    data = input().splitlines()
    Q = int(data[0])
    planting_times = []
    current_time = 0
    harvested_index = 0
    outputs = []
    for i in range(1, Q+1):
        parts = data[i].split()
        if parts[0] == '1':
            bisect.insort(planting_times, current_time)
        elif parts[0] == '2':
            T = int(parts[1])
            current_time += T
        elif parts[0] == '3':
            H = int(parts[1])
            threshold = current_time - H
            index = bisect.bisect_right(planting_times, threshold)
            harvested = index - harvested_index
            outputs.append(str(harvested))
            harvested_index = index
    print('
'.join(outputs))

if __name__ == "__main__":
    main()