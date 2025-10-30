def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    N = int(data[index])
    index += 1
    M = int(data[index])
    index += 1
    
    adjacency = [[] for _ in range(N)]
    for _ in range(M):
        u = int(data[index]) - 1  # 0-indexed
        index += 1
        v = int(data[index]) - 1  # 0-indexed
        index += 1
        adjacency[u].append(v)
        adjacency[v].append(u)
    
    # Build the system of equations
    equations = []
    for v in range(N):
        if len(adjacency[v]) == 0:
            continue
        equation = 0
        for u in adjacency[v]:
            equation ^= (1 << u)
        equations.append(equation)
    
    # Gaussian elimination over GF(2)
    rank = 0
    n_vars = N
    for col in range(n_vars):
        pivot = -1
        for r in range(rank, len(equations)):
            if (equations[r] >> col) & 1:
                pivot = r
                break
        if pivot == -1:
            continue
        # Swap rows
        equations[rank], equations[pivot] = equations[pivot], equations[rank]
        # Eliminate this variable from other equations
        for r in range(len(equations)):
            if r != rank and ((equations[r] >> col) & 1):
                equations[r] ^= equations[rank]
        rank += 1
    
    # Determine free variables
    pivot_vars = set()
    for r in range(rank):
        # Find the position of the leading 1
        for c in range(n_vars):
            if (equations[r] >> c) & 1:
                pivot_vars.add(c)
                break
    free_vars = [c for c in range(n_vars) if c not in pivot_vars]
    
    # Assign values to free variables and compute pivot variables
    # Assign 1 to one free variable and 0 to others
    if len(free_vars) == 0:
        # Only trivial solution, which is all zeros, invalid
        print("No")
        return
    # Assign 1 to the first free variable and 0 to others
    assignment = [0] * n_vars
    for var in free_vars:
        assignment[var] = 0
    if len(free_vars) > 0:
        assignment[free_vars[0]] = 1
    # Compute pivot variables
    for r in range(rank):
        eq = equations[r]
        # Find the pivot variable
        for c in range(n_vars):
            if (eq >> c) & 1:
                pivot_var = c
                break
        # Compute its value
        val = 0
        for c in range(n_vars):
            if c != pivot_var and (eq >> c) & 1:
                val ^= assignment[c]
        assignment[pivot_var] = val
    # Check if all assignments are at least 1
    if all(x >= 1 for x in assignment):
        print("Yes")
        print(' '.join(str(x) for x in assignment))
    else:
        # Try another assignment: set all free variables to 1
        for var in free_vars:
            assignment[var] = 1
        # Compute pivot variables again
        for r in range(rank):
            eq = equations[r]
            # Find the pivot variable
            for c in range(n_vars):
                if (eq >> c) & 1:
                    pivot_var = c
                    break
            # Compute its value
            val = 0
            for c in range(n_vars):
                if c != pivot_var and (eq >> c) & 1:
                    val ^= assignment[c]
            assignment[pivot_var] = val
        if all(x >= 1 for x in assignment):
            print("Yes")
            print(' '.join(str(x) for x in assignment))
        else:
            print("No")

if __name__ == "__main__":
    main()