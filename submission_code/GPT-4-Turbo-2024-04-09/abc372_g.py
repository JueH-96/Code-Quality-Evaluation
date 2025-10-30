import sys
input = sys.stdin.read

def solve():
    data = input().split()
    index = 0
    T = int(data[index])
    index += 1
    results = []
    
    for _ in range(T):
        N = int(data[index])
        index += 1
        A = []
        B = []
        C = []
        
        for _ in range(N):
            A_i = int(data[index])
            B_i = int(data[index + 1])
            C_i = int(data[index + 2])
            A.append(A_i)
            B.append(B_i)
            C.append(C_i)
            index += 3
        
        # To find the maximum x and y that satisfy all conditions:
        # A_i * x + B_i * y < C_i for all i
        # We need to find the maximum x for each i such that:
        # x < (C_i - B_i * y) / A_i for all y > 0
        # and similarly for y:
        # y < (C_i - A_i * x) / B_i for all x > 0
        
        # We need to find the maximum x and y such that all inequalities are satisfied
        max_x = float('inf')
        max_y = float('inf')
        
        for i in range(N):
            # For each constraint, calculate the maximum x and y that could potentially satisfy it
            # We need to consider the smallest C_i / A_i and C_i / B_i across all constraints
            # because any larger x or y would violate at least one constraint.
            max_x = min(max_x, (C[i] - 1) // A[i])
            max_y = min(max_y, (C[i] - 1) // B[i])
        
        # Now we need to count all pairs (x, y) where 1 <= x <= max_x and 1 <= y <= max_y
        # that satisfy all conditions.
        if max_x < 1 or max_y < 1:
            results.append(0)
            continue
        
        count = 0
        for x in range(1, max_x + 1):
            # For each x, determine the maximum y that can work with it
            max_y_for_x = float('inf')
            for i in range(N):
                if B[i] > 0:
                    max_y_for_x = min(max_y_for_x, (C[i] - A[i] * x) // B[i])
            
            if max_y_for_x >= 1:
                count += min(max_y, max_y_for_x)
        
        results.append(count)
    
    for result in results:
        print(result)