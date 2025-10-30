import sys

def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    events = []
    for i in range(N):
        l = int(data[1 + 2*i])
        r = int(data[2 + 2*i + 1])
        events.append((l, 0))  # 0 for start
        events.append((r, 1))  # 1 for end
    # Sort events: first by position, then start before end
    events.sort()
    active_count = 0
    total_intersections = 0
    for event in events:
        if event[1] == 0:
            total_intersections += active_count
            active_count += 1
        else:
            active_count -= 1
    print(total_intersections)

if __name__ == '__main__':
    main()