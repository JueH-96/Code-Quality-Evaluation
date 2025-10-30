from collections import deque

def main():
    S = input().strip()
    T = input().strip()
    
    if S == T:
        print(0)
        return
    
    # Find positions where S differs from T
    diff_positions = [i for i in range(len(S)) if S[i] != T[i]]
    
    # Initialize BFS
    queue = deque([S])
    visited = set()
    visited.add(S)
    parent = {S: None}
    
    found = False
    while queue:
        current = queue.popleft()
        
        # Generate all possible transformations by changing one differing position at a time
        for pos in diff_positions:
            new_string = current[:pos] + T[pos] + current[pos+1:]
            
            if new_string not in visited:
                queue.append(new_string)
                visited.add(new_string)
                parent[new_string] = current
                if new_string == T:
                    found = True
                    break
        if found:
            break
    
    if found:
        # Reconstruct the path from S to T
        path = []
        current = T
        while current:
            path.append(current)
            current = parent[current]
        path.reverse()
        
        M = len(path) - 1
        print(M)
        for step in path[1:]:
            print(step)
    else:
        print(-1)  # Should not happen if S and T have the same length

if __name__ == "__main__":
    main()