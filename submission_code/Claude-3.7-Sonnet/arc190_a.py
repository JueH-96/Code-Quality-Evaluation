# YOUR CODE HERE
def solve(N, M, operations):
    # For each position j, compute the operations that can set it to 1
    position_operations = [[] for _ in range(N + 1)]
    
    for i, (L, R) in enumerate(operations, 1):
        # Operation 1: Set positions [L, R] to 1
        for j in range(L, R + 1):
            position_operations[j].append((i, 1))
        
        # Operation 2: Set positions outside [L, R] to 1
        for j in range(1, L):
            position_operations[j].append((i, 2))
        for j in range(R + 1, N + 1):
            position_operations[j].append((i, 2))
    
    # Check if the goal is achievable
    for j in range(1, N + 1):
        if not position_operations[j]:
            return -1
    
    # Greedy set cover algorithm
    uncovered = set(range(1, N + 1))
    chosen_operations = {}
    
    while uncovered:
        best_operation = None
        best_type = None
        best_coverage = 0
        
        for i in range(1, M + 1):
            if i in chosen_operations:
                continue
            
            for op_type in [1, 2]:
                coverage = 0
                for j in uncovered:
                    if (i, op_type) in position_operations[j]:
                        coverage += 1
                
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_operation = i
                    best_type = op_type
        
        if best_coverage == 0:
            return -1  # This shouldn't happen if the goal is achievable
        
        chosen_operations[best_operation] = best_type
        
        # Update uncovered positions
        for j in list(uncovered):
            if (best_operation, best_type) in position_operations[j]:
                uncovered.remove(j)
    
    # Construct the operation sequence
    operation_sequence = [0] * M
    for i, op_type in chosen_operations.items():
        operation_sequence[i - 1] = op_type
    
    return len(chosen_operations), operation_sequence

N, M = map(int, input().split())
operations = []
for _ in range(M):
    L, R = map(int, input().split())
    operations.append((L, R))

result = solve(N, M, operations)
if result == -1:
    print(-1)
else:
    cost, ops = result
    print(cost)
    print(" ".join(map(str, ops)))