N, Q = map(int, input().split())
S = list(input())

for _ in range(Q):
    query = list(map(int, input().split()))
    
    if query[0] == 1:
        # Flip query
        L, R = query[1]-1, query[2]-1
        for i in range(L, R+1):
            S[i] = '1' if S[i] == '0' else '0'
            
    else:
        # Check query
        L, R = query[1]-1, query[2]-1
        is_good = True
        
        # For single character, it's always good
        if L == R:
            print("Yes")
            continue
            
        # Check if adjacent characters are different
        for i in range(L, R):
            if S[i] == S[i+1]:
                is_good = False
                break
                
        print("Yes" if is_good else "No")