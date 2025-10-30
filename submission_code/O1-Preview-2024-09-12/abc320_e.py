# YOUR CODE HERE
import sys
import threading
import heapq

def main():
    import sys
    import bisect
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    events = []
    for _ in range(M):
        Ti, Wi, Si = map(int, sys.stdin.readline().split())
        heapq.heappush(events, (Ti, 1, Wi, Si))

    set_occupied = set(range(1, N + 1))
    totals = [0] * (N + 1)  # 1-based indexing

    while events:
        time, type_event, *rest = heapq.heappop(events)
        if type_event == 0:  # Return event
            person = rest[0]
            set_occupied.add(person)
        else:  # Noodles event
            Wi, Si = rest
            if set_occupied:
                person = min(set_occupied)
                totals[person] += Wi
                set_occupied.remove(person)
                # Schedule return event
                heapq.heappush(events, (time + Si, 0, person))

    for i in range(1, N + 1):
        print(totals[i])

if __name__ == "__main__":
    threading.Thread(target=main).start()