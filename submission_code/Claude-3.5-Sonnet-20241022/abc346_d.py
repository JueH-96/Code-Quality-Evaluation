N = int(input())
S = input()
C = list(map(int, input().split()))

INF = float('inf')
ans = INF

# Try each possible position i where S[i] == S[i+1]
for i in range(N-1):
    # For each position i, try to make only positions i and i+1 equal
    # and all other adjacent positions different
    cost = 0
    possible = True
    
    # Make copy of string as list for modifications
    T = list(S)
    
    # For each position j < i, make sure T[j] != T[j+1]
    for j in range(i-1, -1, -1):
        if T[j] == T[j+1]:
            # Need to flip one of them
            if j == i-1:
                # Must flip j since we want T[i] == T[i+1]
                T[j] = '1' if T[j] == '0' else '0'
                cost += C[j]
            else:
                # Choose cheaper option to flip
                if C[j] <= C[j+1]:
                    T[j] = '1' if T[j] == '0' else '0'
                    cost += C[j]
                else:
                    T[j+1] = '1' if T[j+1] == '0' else '0'
                    cost += C[j+1]
    
    # For positions i and i+1, make them equal
    if T[i] != T[i+1]:
        # Choose cheaper option to flip
        if C[i] <= C[i+1]:
            T[i] = T[i+1]
            cost += C[i]
        else:
            T[i+1] = T[i]
            cost += C[i+1]
            
    # For each position j > i+1, make sure T[j-1] != T[j]
    for j in range(i+2, N):
        if T[j-1] == T[j]:
            # Need to flip one of them
            if j-1 == i+1:
                # Must flip j since we want T[i] == T[i+1]
                T[j] = '1' if T[j] == '0' else '0'
                cost += C[j]
            else:
                # Choose cheaper option to flip
                if C[j-1] <= C[j]:
                    T[j-1] = '1' if T[j-1] == '0' else '0'
                    cost += C[j-1]
                else:
                    T[j] = '1' if T[j] == '0' else '0'
                    cost += C[j]
    
    # Verify this is a good string
    equal_count = 0
    for j in range(N-1):
        if T[j] == T[j+1]:
            equal_count += 1
    
    if equal_count == 1:
        ans = min(ans, cost)

print(ans)