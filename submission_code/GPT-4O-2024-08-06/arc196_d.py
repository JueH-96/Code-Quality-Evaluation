def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    
    N = int(data[index])
    index += 1
    M = int(data[index])
    index += 1
    Q = int(data[index])
    index += 1
    
    S = []
    T = []
    
    for _ in range(M):
        S.append(int(data[index]) - 1)
        index += 1
        T.append(int(data[index]) - 1)
        index += 1
    
    L = []
    R = []
    
    for _ in range(Q):
        L.append(int(data[index]) - 1)
        index += 1
        R.append(int(data[index]) - 1)
        index += 1
    
    # Initialize the delta array
    delta = [0] * (N + 1)
    
    # Process each person
    for i in range(M):
        start = S[i]
        end = T[i]
        delta[start] += 1
        delta[end] -= 1
    
    # Calculate the prefix sum to get the stamina requirements
    stamina = [0] * (N + 1)
    current_stamina = 0
    for i in range(N):
        current_stamina += delta[i]
        stamina[i] = current_stamina
    
    # Precompute the minimum stamina between each pair of towns
    min_stamina = [float('inf')] * (N + 1)
    current_min = float('inf')
    for i in range(N):
        current_min = min(current_min, stamina[i])
        min_stamina[i] = current_min
    
    # Answer each query
    results = []
    for k in range(Q):
        l = L[k]
        r = R[k]
        
        # Check if all people from L_k to R_k can be satisfied
        possible = True
        for i in range(l, r + 1):
            start = S[i]
            end = T[i]
            if min_stamina[start] <= 0 or min_stamina[end - 1] <= 0:
                possible = False
                break
        
        if possible:
            results.append("Yes")
        else:
            results.append("No")
    
    # Print results
    sys.stdout.write("
".join(results) + "
")