def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return

    it = iter(data)
    n = int(next(it))
    
    # Read arrays:
    A = [int(next(it)) for _ in range(n)]
    B = [int(next(it)) for _ in range(n)]
    C = [int(next(it)) for _ in range(n)]
    
    # The cost for an operation is the sum of C_k for all positions k where A_k is 1 
    # after the flip.
    # When turning a 1 into 0, we effectively remove its contribution from this sum.
    # When turning a 0 into 1, we add its cost.
    # Because operations are sequential, the order in which we flip bits affects the 
    # cost incurred in subsequent operations.
    # Intuition: To minimize total cost, we want to make the ones that we need to vanish (1→0)
    # vanish as early as possible (especially the ones with larger cost–they contribute more) 
    # so that subsequent operations are paid on a reduced sum.
    # For bits that need to be turned from 0 to 1, we want to delay them as long as possible,
    # and among them, it is best to flip the ones with lower cost first.
    
    # Compute initial total cost S from positions where A[i]==1.
    S = 0
    for i in range(n):
        if A[i] == 1:
            S += C[i]
    
    # Identify which positions need flipping.
    ones_to_zero = []  # Here A[i]==1 but B[i]==0.
    zeros_to_one = []  # Here A[i]==0 but B[i]==1.
    for i in range(n):
        if A[i] != B[i]:
            if A[i] == 1:
                ones_to_zero.append(C[i])
            else:
                zeros_to_one.append(C[i])
    
    # For 1->0 flips, sorting in descending order means the expensive contributions are removed early.
    ones_to_zero.sort(reverse=True)
    # For 0->1 flips, sorting in ascending order means we introduce the smaller cost additions first.
    zeros_to_one.sort()
    
    total_cost = 0
    
    # Process the flips from 1 to 0 first.
    for c in ones_to_zero:
        S -= c      # Flipping a 1 to 0 reduces the sum S by its cost.
        total_cost += S  # The operation cost is the current S (after flip).
    
    # Then process flips from 0 to 1.
    for c in zeros_to_one:
        S += c      # Flipping a 0 to 1 increases the sum S by its cost.
        total_cost += S  # Again, the operation cost is the new S.
    
    sys.stdout.write(str(total_cost))
    
if __name__ == '__main__':
    main()