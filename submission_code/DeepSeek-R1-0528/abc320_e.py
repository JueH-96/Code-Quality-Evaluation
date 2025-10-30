import heapq
import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    m = int(data[1])
    events_list = []
    index = 2
    for i in range(m):
        t = int(data[index])
        w = int(data[index+1])
        s = int(data[index+2])
        index += 3
        events_list.append((t, w, s))
    
    available = list(range(1, n+1))
    heapq.heapify(available)
    events = []
    ans = [0] * (n+1)
    
    for event in events_list:
        T, W, S = event
        while events and events[0][0] <= T:
            return_time, person = heapq.heappop(events)
            heapq.heappush(available, person)
        
        if not available:
            continue
            
        p = heapq.heappop(available)
        ans[p] += W
        heapq.heappush(events, (T + S, p))
        
    for i in range(1, n+1):
        print(ans[i])

if __name__ == "__main__":
    main()