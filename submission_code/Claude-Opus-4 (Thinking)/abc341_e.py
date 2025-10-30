# YOUR CODE HERE
N, Q = map(int, input().split())
S = list(input())

for _ in range(Q):
    query_parts = input().split()
    query_type = int(query_parts[0])
    L = int(query_parts[1])
    R = int(query_parts[2])
    
    if query_type == 1:
        # Type 1: Flip characters from L to R (convert to 0-indexed)
        for i in range(L-1, R):
            if S[i] == '0':
                S[i] = '1'
            else:
                S[i] = '0'
    
    else:  # query_type == 2
        # Type 2: Check if substring is good
        is_good = True
        # Check consecutive pairs in the range [L-1, R-1]
        for i in range(L-1, R-1):
            if S[i] == S[i+1]:
                is_good = False
                break
        
        if is_good:
            print("Yes")
        else:
            print("No")