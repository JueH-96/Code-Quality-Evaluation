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
    # Initialize available people
    available = list(range(1, N+1))
    # Min-heap to track when people return
    return_heap = []
    # To store the total noodles each person gets
    noodles = [0] * (N+1)
    for event in events:
        T, W, S = event
        # Process all returns that happen before or at T
        while return_heap and return_heap[0][0] <= T:
            return_time, person = heapq.heappop(return_heap)
            heapq.heappush(available, person)
        if available:
            person = heapq.heappop(available)
            noodles[person] += W
            heapq.heappush(return_heap, (T + S, person))
    for i in range(1, N+1):
        print(noodles[i])

if __name__ == "__main__":
    main()