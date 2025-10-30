def min_cost(N, M, A, B):
    # Create a list of (candy requirement, person index) pairs
    requirements = [(B[i], i) for i in range(M)]
    # Sort in descending order by candy requirement
    requirements.sort(reverse=True)
    
    # Create a list of (price/candy amount, box index) for each box
    boxes = [(A[i], i) for i in range(N)]
    # Sort in ascending order by price/candy amount
    boxes.sort()
    
    total_cost = 0
    assigned_boxes = set()
    
    for candy_req, _ in requirements:
        assigned = False
        for amount, box_idx in boxes:
            if amount >= candy_req and box_idx not in assigned_boxes:
                total_cost += amount
                assigned_boxes.add(box_idx)
                assigned = True
                break
        
        if not assigned:
            return -1  # Impossible to satisfy the condition
    
    return total_cost

# Read input
N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Compute and output the result
print(min_cost(N, M, A, B))