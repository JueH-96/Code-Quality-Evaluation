# YOUR CODE HERE
N = int(input())
A = list(map(int, input().split()))
B = list(map(int, input().split()))
Q = int(input())

for _ in range(Q):
    query = list(map(int, input().split()))
    
    if query[0] == 1:
        # Type 1: Replace A[i-1] with x (1-indexed to 0-indexed)
        i, x = query[1], query[2]
        A[i-1] = x
    
    elif query[0] == 2:
        # Type 2: Replace B[i-1] with x (1-indexed to 0-indexed)
        i, x = query[1], query[2]
        B[i-1] = x
    
    else:  # query[0] == 3
        # Type 3: Find maximum value
        l, r = query[1], query[2]
        
        # Start with v = 0
        v = 0
        
        # Process from position l to r (1-indexed)
        for i in range(l-1, r):
            # Choose between adding A[i] or multiplying by B[i]
            option1 = v + A[i]
            option2 = v * B[i]
            v = max(option1, option2)
        
        print(v)