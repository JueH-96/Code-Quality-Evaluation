import sys
from heapq import heappop, heappush

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    trains = []
    for i in range(M):
        l = int(data[2 + 6*i])
        d = int(data[3 + 6*i])
        k = int(data[4 + 6*i])
        c = int(data[5 + 6*i])
        A = int(data[6 + 6*i])
        B = int(data[7 + 6*i])
        for j in range(k):
            trains.append((l + j*d, c, A, B))
    
    trains.sort()
    
    # Initialize the latest arrival times
    latest_arrival = [-float('inf')] * (N + 1)
    latest_arrival[N] = float('inf')
    
    # Priority queue to keep track of the latest arrival times
    pq = []
    
    for t, c, A, B in trains:
        while pq and pq[0][0] < t:
            _, station = heappop(pq)
            latest_arrival[station] = -float('inf')
        
        if latest_arrival[A] != -float('inf'):
            new_arrival_time = latest_arrival[A] + c
            if new_arrival_time > latest_arrival[B]:
                latest_arrival[B] = new_arrival_time
                heappush(pq, (new_arrival_time, B))
    
    for i in range(1, N):
        if latest_arrival[i] == -float('inf'):
            print("Unreachable")
        else:
            print(latest_arrival[i])

if __name__ == "__main__":
    main()