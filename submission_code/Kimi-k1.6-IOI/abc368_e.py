import sys
import heapq
import bisect
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    X1 = int(input[ptr])
    ptr += 1

    trains = []
    for _ in range(M):
        A = int(input[ptr])
        ptr += 1
        B = int(input[ptr])
        ptr += 1
        S = int(input[ptr])
        ptr += 1
        T = int(input[ptr])
        ptr += 1
        trains.append((A, B, S, T))
    
    departures = defaultdict(list)
    for j in range(M):
        A_j = trains[j][0]
        S_j = trains[j][2]
        departures[A_j].append((S_j, j))
    
    # Preprocess departures to have sorted Sj lists and the corresponding j's
    for city in list(departures.keys()):
        departures[city].sort()
        Sj_list = [t[0] for t in departures[city]]
        departures[city] = (Sj_list, departures[city])
    
    adj = [[] for _ in range(M)]
    for i in range(M):
        _, B_i, _, T_i = trains[i]
        arrival_city = B_i
        if arrival_city not in departures:
            continue
        Sj_list, list_of_tuples = departures[arrival_city]
        idx = bisect.bisect_left(Sj_list, T_i)
        for t in list_of_tuples[idx:]:
            S_j, j = t
            weight = S_j - T_i
            adj[i].append((j, weight))
    
    INF = float('inf')
    distance = [INF] * M
    distance[0] = 0
    heap = [(0, 0)]
    
    while heap:
        dist_u, u = heapq.heappop(heap)
        if dist_u > distance[u]:
            continue
        for v, w in adj[u]:
            if distance[v] > dist_u + w:
                distance[v] = dist_u + w
                heapq.heappush(heap, (distance[v], v))
    
    X = []
    for j in range(1, M):
        val = X1 - distance[j]
        X.append(max(0, val))
    
    print(' '.join(map(str, X)))

if __name__ == '__main__':
    main()