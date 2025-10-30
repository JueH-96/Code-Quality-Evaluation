# YOUR CODE HERE
import sys
import heapq
from collections import defaultdict

def solve():
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    N = int(data[index])
    index += 1
    M = int(data[index])
    index += 1
    
    trains = defaultdict(list)
    
    for _ in range(M):
        l_i = int(data[index])
        index += 1
        d_i = int(data[index])
        index += 1
        k_i = int(data[index])
        index += 1
        c_i = int(data[index])
        index += 1
        A_i = int(data[index])
        index += 1
        B_i = int(data[index])
        index += 1
        
        for j in range(k_i):
            departure_time = l_i + j * d_i
            arrival_time = departure_time + c_i
            trains[A_i].append((departure_time, arrival_time, B_i))
    
    results = []
    
    for start in range(1, N):
        # Max-heap for latest arrival time
        max_heap = []
        # Dictionary to store the latest arrival time at each station
        latest_arrival = defaultdict(lambda: -float('inf'))
        
        # Start from the station `start`
        for departure_time, arrival_time, dest in trains[start]:
            heapq.heappush(max_heap, (-arrival_time, dest))
        
        while max_heap:
            neg_arrival_time, current_station = heapq.heappop(max_heap)
            arrival_time = -neg_arrival_time
            
            if current_station == N:
                results.append(arrival_time)
                break
            
            if arrival_time > latest_arrival[current_station]:
                latest_arrival[current_station] = arrival_time
                for dep_time, arr_time, dest in trains[current_station]:
                    if dep_time >= arrival_time:
                        heapq.heappush(max_heap, (-arr_time, dest))
        
        else:
            results.append("Unreachable")
    
    for result in results:
        print(result)