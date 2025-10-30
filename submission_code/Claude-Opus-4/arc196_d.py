def solve():
    N, M, Q = map(int, input().split())
    
    people = []
    for i in range(M):
        s, t = map(int, input().split())
        people.append((s-1, t-1))  # 0-indexed
    
    queries = []
    for i in range(Q):
        l, r = map(int, input().split())
        queries.append((l-1, r-1))  # 0-indexed
    
    # For each query
    for l, r in queries:
        # Check if people[l:r+1] can be satisfied
        can_satisfy = True
        
        # Build constraint system
        # We'll use a graph-based approach to check consistency
        # Each road strength can be represented as a variable
        # We need to check if the constraints form a consistent system
        
        # Create adjacency list for constraint graph
        from collections import defaultdict, deque
        
        # For each person, we have constraints
        constraints = []
        
        for i in range(l, r+1):
            s, t = people[i]
            if s > t:
                s, t = t, s
            constraints.append((s, t))
        
        # Check for conflicts
        # Two key observations:
        # 1. If two paths overlap, they must be compatible
        # 2. If a path is contained in another, specific conditions must hold
        
        # Sort constraints by start position
        constraints.sort()
        
        # Check for impossible cases
        for i in range(len(constraints)):
            for j in range(i+1, len(constraints)):
                s1, t1 = constraints[i]
                s2, t2 = constraints[j]
                
                # If paths overlap
                if s1 <= s2 < t1:
                    # Check if they're compatible
                    # If s2 < t1 <= t2: partial overlap
                    # If t2 <= t1: s2,t2 is contained in s1,t1
                    
                    if t2 <= t1:
                        # Path 2 is contained in path 1
                        # This means person going s2->t2 must have positive stamina at s2 and t2
                        # But they start with 0 at s2 and end with 0 at t2
                        # This is impossible if s2 > s1 (they'd need positive at s2)
                        if s2 > s1:
                            can_satisfy = False
                            break
                    else:
                        # Partial overlap
                        # Need to check if the overlapping part can satisfy both
                        # This gets complex, but key insight:
                        # If two people share a subpath, the sum on that subpath must work for both
                        pass
            
            if not can_satisfy:
                break
        
        # Additional check: if multiple people use exact same path
        path_count = defaultdict(int)
        for s, t in constraints:
            path_count[(s, t)] += 1
            if path_count[(s, t)] > 1:
                can_satisfy = False
                break
        
        # More sophisticated check for overlapping paths
        if can_satisfy:
            # Build interval tree to check overlaps more carefully
            events = []
            for idx, (s, t) in enumerate(constraints):
                events.append((s, 1, idx))  # start
                events.append((t, -1, idx))  # end
            
            events.sort()
            active = set()
            
            for pos, typ, idx in events:
                if typ == 1:
                    # Check if adding this path creates conflict
                    for other_idx in active:
                        # Both paths are active at this position
                        s1, t1 = constraints[idx]
                        s2, t2 = constraints[other_idx]
                        
                        # If one contains the other's start or end (but not both)
                        if (s1 < s2 < t1 and t2 > t1) or (s2 < s1 < t2 and t1 > t2):
                            can_satisfy = False
                            break
                    
                    active.add(idx)
                else:
                    active.remove(idx)
                
                if not can_satisfy:
                    break
        
        print("Yes" if can_satisfy else "No")

solve()