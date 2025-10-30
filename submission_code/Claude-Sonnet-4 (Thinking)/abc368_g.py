N = int(input())
A = list(map(int, input().split()))
B = list(map(int, input().split()))
Q = int(input())

for _ in range(Q):
    query = list(map(int, input().split()))
    
    if query[0] == 1:
        # Type 1: Replace A[i] with x
        i, x = query[1] - 1, query[2]  # Convert to 0-indexed
        A[i] = x
    elif query[0] == 2:
        # Type 2: Replace B[i] with x
        i, x = query[1] - 1, query[2]  # Convert to 0-indexed
        B[i] = x
    else:
        # Type 3: Solve the problem for range [l, r]
        l, r = query[1] - 1, query[2] - 1  # Convert to 0-indexed
        
        v = 0
        for i in range(l, r + 1):
            choice1 = v + A[i]
            choice2 = v * B[i]
            v = max(choice1, choice2)
        
        print(v)