import sys
import heapq

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    M = int(data[idx+1])
    idx += 2
    events = []
    for _ in range(M):
        T = int(data[idx])
        W = int(data[idx+1])
        S = int(data[idx+2])
        events.append((T, W, S))
        idx += 3
    # Initialize the available people
    available = list(range(1, N+1))
    # Min-heap to keep track of when people return
    returning = []
    # To store the total noodles each person gets
    noodles = [0] * (N+1)
    for event in events:
        T, W, S = event
        # First, process all returning people whose return time is <= T
        while returning and returning[0][0] <= T:
            time, person = heapq.heappop(returning)
            heapq.heappush(available, person)
        if available:
            person = heapq.heappop(available)
            noodles[person] += W
            heapq.heappush(returning, (T + S, person))
    for i in range(1, N+1):
        print(noodles[i])

if __name__ == "__main__":
    main()