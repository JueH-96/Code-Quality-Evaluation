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
        
        # Calculate the maximum x and y bounds
        max_x = float('inf')
        for i in range(N):
            max_x = min(max_x, C[i] // A[i])
        
        count = 0
        
        for x in range(1, max_x):
            max_y = float('inf')
            for i in range(N):
                if A[i] * x < C[i]:
                    max_y = min(max_y, (C[i] - A[i] * x) // B[i])
                else:
                    max_y = 0
                    break
            
            if max_y > 0:
                count += max_y
        
        results.append(count)
    
    for result in results:
        print(result)