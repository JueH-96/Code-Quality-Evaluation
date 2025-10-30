from collections import deque

N, X, Y = map(int, input().split())
S = input().strip()
T = input().strip()

if S == T:
    print("Yes")
else:
    # BFS to find if we can transform S to T
    queue = deque([S])
    visited = {S}
    found = False
    
    while queue and not found:
        current = queue.popleft()
        
        # Try Operation A: X zeros + Y ones -> Y ones + X zeros
        for i in range(N - X - Y + 1):
            if (all(current[i+j] == '0' for j in range(X)) and 
                all(current[i+X+j] == '1' for j in range(Y))):
                
                new_s = list(current)
                for j in range(Y):
                    new_s[i+j] = '1'
                for j in range(X):
                    new_s[i+Y+j] = '0'
                new_s = ''.join(new_s)
                
                if new_s == T:
                    found = True
                    break
                
                if new_s not in visited:
                    visited.add(new_s)
                    queue.append(new_s)
        
        if found:
            break
            
        # Try Operation B: Y ones + X zeros -> X zeros + Y ones
        for i in range(N - X - Y + 1):
            if (all(current[i+j] == '1' for j in range(Y)) and 
                all(current[i+Y+j] == '0' for j in range(X))):
                
                new_s = list(current)
                for j in range(X):
                    new_s[i+j] = '0'
                for j in range(Y):
                    new_s[i+X+j] = '1'
                new_s = ''.join(new_s)
                
                if new_s == T:
                    found = True
                    break
                
                if new_s not in visited:
                    visited.add(new_s)
                    queue.append(new_s)
    
    print("Yes" if found else "No")