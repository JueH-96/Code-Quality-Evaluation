import heapq
import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1

    events = []
    for _ in range(M):
        T = int(input[ptr])
        ptr += 1
        W = int(input[ptr])
        ptr += 1
        S = int(input[ptr])
        ptr += 1
        events.append((T, W, S))
    
    available = list(range(1, N + 1))
    heapq.heapify(available)
    
    unavailable = []
    ans = [0] * (N + 1)
    
    for T, W, S in events:
        # Process all unavailable people who can return by current time T
        while unavailable and unavailable[0][0] <= T:
            return_time, person = heapq.heappop(unavailable)
            heapq.heappush(available, person)
        
        # Assign noodles if someone is available
        if available:
            current_person = heapq.heappop(available)
            ans[current_person] += W
            heapq.heappush(unavailable, (T + S, current_person))
    
    for i in range(1, N + 1):
        print(ans[i])

if __name__ == '__main__':
    main()