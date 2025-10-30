def can_satisfy_requirements(N, paths):
    # For each path, we need stamina to be 0 at start and end
    # and positive at intermediate points
    
    # Check if paths are conflicting
    # Create a graph of dependencies between paths
    adj = [[] for _ in range(len(paths))]
    
    for i in range(len(paths)):
        s1, t1 = paths[i]
        # Make s1 < t1 for consistent comparison
        if s1 > t1:
            s1, t1 = t1, s1
            
        for j in range(i + 1, len(paths)):
            s2, t2 = paths[j]
            # Make s2 < t2 for consistent comparison
            if s2 > t2:
                s2, t2 = t2, s2
                
            # Check if paths overlap and create conflict
            if not (t1 < s2 or t2 < s1):
                # If paths share more than one point and have different directions
                # they cannot be satisfied simultaneously
                if (s1 < s2 < t1 and s1 < t2 < t1) or (s2 < s1 < t2 and s2 < t1 < t2):
                    adj[i].append(j)
                    adj[j].append(i)
    
    # If any two adjacent paths in the graph are selected, return False
    def is_independent_set(selected):
        for i in selected:
            for j in adj[i]:
                if j in selected:
                    return False
        return True
    
    return is_independent_set

def main():
    # Read input
    N, M, Q = map(int, input().split())
    paths = []
    for _ in range(M):
        s, t = map(int, input().split())
        paths.append((s, t))
    
    # Process queries
    for _ in range(Q):
        L, R = map(int, input().split())
        # Convert to 0-based indexing
        L -= 1
        R -= 1
        
        # Check if paths from L to R can be satisfied
        selected = set(range(L, R + 1))
        if can_satisfy_requirements(N, [paths[i] for i in selected]):
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()