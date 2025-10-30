import heapq

N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

def circular_distance(a, b, m):
    return min(abs(a - b), m - abs(a - b))

def heuristic(seq):
    return sum(circular_distance(seq[i], B[i], M) for i in range(N))

def get_neighbors(seq):
    neighbors = []
    for i in range(len(seq)):
        for delta in [1, -1]:
            new_val = (seq[i] + delta) % M
            # Check if the new value maintains the good sequence property
            if (i == 0 or new_val != seq[i - 1]) and (i == N - 1 or new_val != seq[i + 1]):
                new_seq = seq[:]
                new_seq[i] = new_val
                neighbors.append(tuple(new_seq))
    return neighbors

if A == B:
    print(0)
else:
    heap = [(heuristic(A), 0, tuple(A))]
    visited = set()
    
    while heap:
        _, operations, current_seq = heapq.heappop(heap)
        
        if current_seq in visited:
            continue
        
        visited.add(current_seq)
        
        if list(current_seq) == B:
            print(operations)
            break
        
        for neighbor in get_neighbors(list(current_seq)):
            if neighbor not in visited:
                new_operations = operations + 1
                priority = new_operations + heuristic(list(neighbor))
                heapq.heappush(heap, (priority, new_operations, neighbor))
    else:
        print(-1)