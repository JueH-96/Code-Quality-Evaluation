import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
D = list(map(int, data[1:N+1]))
L = list(map(int, data[N+1:2*N+1]))
C = list(map(int, data[2*N+1:3*N+1]))
K = list(map(int, data[3*N+1:]))

def can_monitor(section, sensor_length):
    return section <= sensor_length

def min_cost_to_monitor_sections(N, D, L, C, K):
    from collections import defaultdict
    from heapq import heappop, heappush
    
    # Create a list of all sections to be monitored
    sections = []
    for i in range(N):
        sections.extend([D[i]] * (D[i] // L[i] + (1 if D[i] % L[i] != 0 else 0)))
    
    # Sort sections by length in descending order
    sections.sort(reverse=True)
    
    # Priority queue to store the cost of using each type of sensor
    pq = []
    for j in range(2):
        heappush(pq, (C[j], K[j]))
    
    total_cost = 0
    for section in sections:
        if pq:
            cost, count = heappop(pq)
            if count > 0:
                total_cost += cost
                count -= 1
                heappush(pq, (cost, count))
            else:
                heappush(pq, (cost, count))
        else:
            return -1
    
    return total_cost

print(min_cost_to_monitor_sections(N, D, L, C, K))