import sys
from collections import defaultdict, deque

def solve():
    input = sys.stdin.read
    data = input().split()
    
    N, M, X1 = map(int, data[0:3])
    idx = 3
    trains = []
    for _ in range(M):
        A, B, S, T = map(int, data[idx:idx+4])
        trains.append((A-1, B-1, S, T))
        idx += 4
    
    # Create a graph where each node represents a train and edges represent transfer possibilities
    graph = defaultdict(list)
    for i in range(M):
        for j in range(M):
            if trains[i][1] == trains[j][0] and trains[i][3] <= trains[j][2]:
                graph[i].append(j)
    
    # Initialize delays with X1 for the first train and 0 for others
    delays = [0] * M
    delays[0] = X1
    
    # Use a queue to process trains in topological order
    queue = deque([0])
    while queue:
        current = queue.popleft()
        for next_train in graph[current]:
            required_delay = max(0, trains[next_train][2] - trains[current][3] + delays[current] - delays[next_train])
            if required_delay > 0:
                delays[next_train] += required_delay
                queue.append(next_train)
    
    # Output the delays for trains 2 to M
    print(' '.join(map(str, delays[1:])))

solve()