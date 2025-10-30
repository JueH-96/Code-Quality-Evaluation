def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    
    # Read sequences A, B and costs C.
    A = [int(next(it)) for _ in range(n)]
    B = [int(next(it)) for _ in range(n)]
    C = [int(next(it)) for _ in range(n)]
    
    # Initially, the state’s "cost sum" is calculated by summing C_i for indices with A_i == 1.
    current_sum = 0
    for i in range(n):
        if A[i] == 1:
            current_sum += C[i]
    
    # We need to flip some positions.
    # For positions where A[i] = 1 and B[i] = 0, we perform a 1 -> 0 flip ("removals").
    # For positions where A[i] = 0 and B[i] = 1, we perform a 0 -> 1 flip ("additions").
    removals = []
    additions = []
    for i in range(n):
        if A[i] == 1 and B[i] == 0:
            removals.append(C[i])
        if A[i] == 0 and B[i] == 1:
            additions.append(C[i])
            
    # The cost of an operation is the sum over ones after the flip.
    # To minimize the overall cost we want to:
    # - Remove 1's (that should be 0 in target) as early as possible, especially those with expensive cost.
    #   Hence, for removals, we sort in descending order (high cost removed first).
    # - For additions, since flipping from 0->1 increases the ones sum and therefore increases future operation costs,
    #   we want to delay expensive additions. We sort additions in ascending order (adding cheaper ones first).
    removals.sort(reverse=True)
    additions.sort()
    
    total_cost = 0
    # Process removal operations.
    for cost_val in removals:
        # When we flip a 1 to 0, the new state no longer has this cost.
        # Thus, the cost paid for this operation is the current ones sum after removal.
        op_cost = current_sum - cost_val
        total_cost += op_cost
        current_sum -= cost_val  # update state: removed 1.
    
    # Process addition operations.
    for cost_val in additions:
        # When adding a 1 (from 0 to 1), the new state's ones sum increases by cost_val.
        # The cost paid for the operation is the new current sum.
        op_cost = current_sum + cost_val
        total_cost += op_cost
        current_sum += cost_val  # update state: added 1.
    
    sys.stdout.write(str(total_cost))

if __name__ == '__main__':
    main()