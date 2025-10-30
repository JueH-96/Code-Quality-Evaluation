import heapq
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1

    ans = [0] * (N + 1)
    available = list(range(1, N + 1))
    heapq.heapify(available)

    return_events = []

    for _ in range(M):
        T = int(data[idx])
        W = int(data[idx + 1])
        S = int(data[idx + 2])
        idx += 3

        # Process all return events up to current time T
        while return_events and return_events[0][0] <= T:
            time, person = heapq.heappop(return_events)
            heapq.heappush(available, person)

        # Assign noodles if possible
        if available:
            person = heapq.heappop(available)
            ans[person] += W
            heapq.heappush(return_events, (T + S, person))

    # Output the results
    for i in range(1, N + 1):
        print(ans[i])

if __name__ == "__main__":
    main()