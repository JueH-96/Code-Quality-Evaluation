def solve_confused_villagers(n, testimonies):
    if not testimonies:
        return "0" * n  # M=0 case
    
    # Extract all unique villagers mentioned in testimonies
    unique_villagers = set()
    for A, B, C in testimonies:
        unique_villagers.add(A)
        unique_villagers.add(B)
    
    villagers_list = sorted(list(unique_villagers))
    max_combinations = 2 ** len(villagers_list)
    
    # Try all possible combinations of confused villagers
    for i in range(max_combinations):
        confused = [0] * (n + 1)  # 1-indexed
        
        # Assign confused status to villagers
        for j, villager in enumerate(villagers_list):
            if (i >> j) & 1:
                confused[villager] = 1
        
        if is_consistent(n, testimonies, confused):
            return "".join(map(str, confused[1:]))
    
    return "-1"  # No consistent assignment exists

def is_consistent(n, testimonies, confused):
    """
    Check if the testimonies are consistent for the given confused vector
    using 2-SAT representation
    """
    # Build a 2-SAT implication graph
    graph = {}
    for i in range(1, n+1):
        graph[i] = []       # i is honest
        graph[-i] = []      # i is a liar
    
    for A, B, C in testimonies:
        # Determine truthfulness based on confused status
        # If confused, they do the opposite of what they normally do
        # A honest + not confused = tells truth
        # A liar + not confused = tells lies
        # A honest + confused = tells lies
        # A liar + confused = tells truth
        truthful_if_honest = not confused[A]
        truthful_if_liar = confused[A]
        
        if C == 0:  # A says B is honest
            if truthful_if_honest:
                # If A is honest and truthful, B is honest
                graph[-A].append(B)   # not A or B
                graph[-B].append(-A)  # not B or not A
            else:
                # If A is honest and lies, B is a liar
                graph[-A].append(-B)  # not A or not B
                graph[B].append(-A)   # B or not A
            
            if truthful_if_liar:
                # If A is a liar and truthful, B is honest
                graph[A].append(B)    # A or B
                graph[-B].append(A)   # not B or A
            else:
                # If A is a liar and lies, B is a liar
                graph[A].append(-B)   # A or not B
                graph[B].append(A)    # B or A
        else:  # C == 1, A says B is a liar
            if truthful_if_honest:
                # If A is honest and truthful, B is a liar
                graph[-A].append(-B)  # not A or not B
                graph[B].append(-A)   # B or not A
            else:
                # If A is honest and lies, B is honest
                graph[-A].append(B)   # not A or B
                graph[-B].append(-A)  # not B or not A
            
            if truthful_if_liar:
                # If A is a liar and truthful, B is a liar
                graph[A].append(-B)   # A or not B
                graph[B].append(A)    # B or A
            else:
                # If A is a liar and lies, B is honest
                graph[A].append(B)    # A or B
                graph[-B].append(A)   # not B or A
    
    # Check if the 2-SAT formula is satisfiable
    # If any villager must be both honest and a liar, the formula is unsatisfiable
    for v in range(1, n+1):
        if is_reachable(graph, v, -v) and is_reachable(graph, -v, v):
            return False
    
    return True

def is_reachable(graph, start, end):
    """
    Check if there's a path from start to end in the graph.
    """
    visited = set()
    stack = [start]
    visited.add(start)
    
    while stack:
        node = stack.pop()
        if node == end:
            return True
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    
    return False

# Read input
n, m = map(int, input().split())
testimonies = []
for _ in range(m):
    A, B, C = map(int, input().split())
    testimonies.append((A, B, C))

# Solve and print result
print(solve_confused_villagers(n, testimonies))