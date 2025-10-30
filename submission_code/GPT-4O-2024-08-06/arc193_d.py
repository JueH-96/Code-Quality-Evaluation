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
        
        # Precompute nearest left and right positions with A[i] = '1'
        left_nearest = [-1] * N
        right_nearest = [-1] * N
        
        # Fill left_nearest
        last_one = -1
        for i in range(N):
            if A[i] == '1':
                last_one = i
            left_nearest[i] = last_one
        
        # Fill right_nearest
        last_one = -1
        for i in range(N-1, -1, -1):
            if A[i] == '1':
                last_one = i
            right_nearest[i] = last_one
        
        # Calculate the minimum number of operations
        min_operations = 0
        possible = True
        
        for i in range(N):
            if B[i] == '1':
                if left_nearest[i] == -1 and right_nearest[i] == -1:
                    possible = False
                    break
                left_dist = float('inf') if left_nearest[i] == -1 else i - left_nearest[i]
                right_dist = float('inf') if right_nearest[i] == -1 else right_nearest[i] - i
                min_operations += min(left_dist, right_dist)
        
        if possible:
            results.append(str(min_operations))
        else:
            results.append('-1')
    
    print("
".join(results))