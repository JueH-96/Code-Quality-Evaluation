from collections import deque

def main():
    # Read input
    N, M = map(int, input().split())
    A, B = [], []
    for _ in range(M):
        a, b = map(int, input().split())
        A.append(a)
        B.append(b)
    
    # Solve the problem
    result = solve(N, M, A, B)
    
    # Print output
    print(result)

def solve(N, M, A, B):
    # Build adjacency list
    graph = [[] for _ in range(N + 1)]
    for a, b in zip(A, B):
        graph[a].append(b)
        graph[b].append(a)
    
    # Set of all friendships
    friendships = set()
    for a, b in zip(A, B):
        friendships.add((min(a, b), max(a, b)))
    
    count = 0
    operations_queue = deque()
    pending_operations = set()
    
    # Find initial open triads
    for y in range(1, N + 1):
        neighbors = graph[y]
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                x, z = neighbors[i], neighbors[j]
                pair = (min(x, z), max(x, z))
                if pair not in friendships and pair not in pending_operations:
                    operations_queue.append(pair)
                    pending_operations.add(pair)
    
    while operations_queue:
        pair = operations_queue.popleft()
        pending_operations.remove(pair)
        
        if pair in friendships:
            continue
        
        x, z = pair
        friendships.add(pair)
        count += 1
        
        # Update the graph
        graph[x].append(z)
        graph[z].append(x)
        
        # Find new open triads
        for w in graph[x]:
            if w != z:
                new_pair = (min(w, z), max(w, z))
                if new_pair not in friendships and new_pair not in pending_operations:
                    operations_queue.append(new_pair)
                    pending_operations.add(new_pair)
        
        for w in graph[z]:
            if w != x:
                new_pair = (min(w, x), max(w, x))
                if new_pair not in friendships and new_pair not in pending_operations:
                    operations_queue.append(new_pair)
                    pending_operations.add(new_pair)
    
    return count

if __name__ == "__main__":
    main()