import sys
from heapq import heappop, heappush

def main():
    input = sys.stdin.read
    data = input().split()

    index = 0
    N = int(data[index])
    index += 1
    K = int(data[index])
    index += 1

    events = []

    for _ in range(N):
        a = int(data[index])
        index += 1
        b = int(data[index])
        index += 2
        events.append((a, b))
        events.append((a + b, -b))

    events.sort()

    current_pills = 0
    min_day = float('inf')

    for day, pills in events:
        current_pills += pills
        if current_pills <= K:
            min_day = min(min_day, day)

    print(min_day)

if __name__ == "__main__":
    main()