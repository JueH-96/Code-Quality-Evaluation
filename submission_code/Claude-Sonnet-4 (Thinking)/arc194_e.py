from collections import deque

def solve():
    N, X, Y = map(int, input().split())
    S = input().strip()
    T = input().strip()
    
    if S == T:
        return "Yes"
    
    if N < X + Y:
        return "No"
    
    # Check if S and T have the same number of 0s and 1s
    if S.count('0') != T.count('0'):
        return "No"
    
    visited = set()
    queue = deque([S])
    visited.add(S)
    
    while queue:
        current = queue.popleft()
        
        # Try Operation A: "0^X 1^Y" -> "1^Y 0^X"
        for i in range(N - X - Y + 1):
            if (current[i:i+X] == '0' * X and current[i+X:i+X+Y] == '1' * Y):
                new_s = list(current)
                for j in range(i, i+Y):
                    new_s[j] = '1'
                for j in range(i+Y, i+Y+X):
                    new_s[j] = '0'
                new_s = ''.join(new_s)
                
                if new_s == T:
                    return "Yes"
                if new_s not in visited:
                    visited.add(new_s)
                    queue.append(new_s)
        
        # Try Operation B: "1^Y 0^X" -> "0^X 1^Y"
        for i in range(N - X - Y + 1):
            if (current[i:i+Y] == '1' * Y and current[i+Y:i+Y+X] == '0' * X):
                new_s = list(current)
                for j in range(i, i+X):
                    new_s[j] = '0'
                for j in range(i+X, i+X+Y):
                    new_s[j] = '1'
                new_s = ''.join(new_s)
                
                if new_s == T:
                    return "Yes"
                if new_s not in visited:
                    visited.add(new_s)
                    queue.append(new_s)
    
    return "No"

print(solve())