def max_servings(N, Q, A, B):
    """Find the maximum total number of servings."""
    points = [(0, 0)]  # Start with the origin
    
    # Add points with a = 0
    for i in range(N):
        if B[i] > 0:
            points.append((0, Q[i] // B[i]))
    
    # Add points with b = 0
    for i in range(N):
        if A[i] > 0:
            points.append((Q[i] // A[i], 0))
    
    # Add intersection points of two constraints
    for i in range(N):
        for j in range(i+1, N):
            if A[i] == 0 and A[j] == 0:
                continue
            if B[i] == 0 and B[j] == 0:
                continue
            
            det = A[i] * B[j] - A[j] * B[i]
            if det == 0:  # Parallel lines
                continue
            
            # Solve the system of equations
            a_num = Q[i] * B[j] - Q[j] * B[i]
            b_num = A[i] * Q[j] - A[j] * Q[i]
            
            # Check if the solution is an integer
            if a_num % abs(det) == 0 and b_num % abs(det) == 0:
                a = a_num / det
                b = b_num / det
                
                if a >= 0 and b >= 0:
                    points.append((int(a), int(b)))
    
    # Check each point against all constraints
    max_total = 0
    for a, b in points:
        valid = True
        for i in range(N):
            if a * A[i] + b * B[i] > Q[i]:
                valid = False
                break
        
        if valid:
            max_total = max(max_total, a + b)
    
    return max_total

# Read input
N = int(input())
Q = list(map(int, input().split()))
A = list(map(int, input().split()))
B = list(map(int, input().split()))

print(max_servings(N, Q, A, B))