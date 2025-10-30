def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    T = int(data[index])
    index += 1
    results = []
    
    for _ in range(T):
        N = int(data[index])
        index += 1
        A = data[index]
        index += 1
        B = data[index]
        index += 1
        
        # We need to find the minimum number of moves to make A transform into B
        # where B_i = 1 means there must be at least one piece in square i
        
        # We can use a two-pointer technique to simulate the movement of pieces
        # to match the configuration B from configuration A
        
        # Leftmost and rightmost positions where pieces need to be
        left_needed = next(i for i in range(N) if B[i] == '1')
        right_needed = next(i for i in range(N-1, -1, -1) if B[i] == '1')
        
        # Leftmost and rightmost positions where pieces are initially
        left_available = next(i for i in range(N) if A[i] == '1')
        right_available = next(i for i in range(N-1, -1, -1) if A[i] == '1')
        
        # If the needed span is outside the available span, it's impossible
        if left_needed < left_available or right_needed > right_available:
            results.append(-1)
            continue
        
        # Calculate the minimum moves needed
        # We need to cover all positions from left_needed to right_needed
        # with pieces that are initially from left_available to right_available
        
        # We will use a sliding window to determine the minimum shifts needed
        # to cover all required '1's in B with available '1's in A
        
        # We need to ensure that every position i where B[i] == '1' has at least one '1' from A in range
        # after some operations
        
        # We will simulate the process of moving pieces to the left or right to cover all required positions
        
        # We will count how many operations are needed to move all pieces to cover all required positions
        min_operations = 0
        j = 0  # Pointer for A
        
        # We need to cover each position i where B[i] == '1'
        for i in range(N):
            if B[i] == '1':
                # Move j to the next '1' in A if necessary
                while j < N and A[j] == '0':
                    j += 1
                if j < N:
                    # Calculate the cost to cover position i using position j
                    min_operations = max(min_operations, abs(i - j))
                j += 1  # Move j to the next position for the next required '1'
        
        results.append(min_operations)
    
    # Print all results
    for result in results:
        print(result)