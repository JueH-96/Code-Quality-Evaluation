# YOUR CODE HERE
import sys
from collections import defaultdict, deque

def main():
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    N = int(data[index])
    index += 1
    M = int(data[index])
    index += 1
    
    sets = []
    element_to_sets = defaultdict(list)
    
    for i in range(N):
        A_i = int(data[index])
        index += 1
        S_i = list(map(int, data[index:index + A_i]))
        index += A_i
        sets.append(S_i)
        for element in S_i:
            element_to_sets[element].append(i)
    
    # BFS to find the shortest path from any set containing 1 to any set containing M
    start_sets = element_to_sets[1]
    end_sets = set(element_to_sets[M])
    
    if not start_sets or not end_sets:
        print(-1)
        return
    
    queue = deque()
    visited = [False] * N
    
    for start in start_sets:
        queue.append((start, 0))
        visited[start] = True
    
    while queue:
        current_set, operations = queue.popleft()
        
        if current_set in end_sets:
            print(operations)
            return
        
        # Check all elements in the current set
        for element in sets[current_set]:
            # Check all sets that contain this element
            for neighbor_set in element_to_sets[element]:
                if not visited[neighbor_set]:
                    visited[neighbor_set] = True
                    queue.append((neighbor_set, operations + 1))
    
    print(-1)