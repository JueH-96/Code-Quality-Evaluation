def solve():
    t = int(input())
    for _ in range(t):
        S = input().strip()
        X = input().strip()
        Y = input().strip()
        
        n0_X = X.count('0')
        n1_X = X.count('1')
        n0_Y = Y.count('0')
        n1_Y = Y.count('1')
        
        a = n0_X - n0_Y
        b = n1_Y - n1_X
        
        if a == 0 and b == 0:
            # Any length of T works, so check if X == Y
            print("Yes" if X == Y else "No")
        elif a == 0 and b != 0:
            # T must be empty
            if can_match(S, 0, X, Y):
                print("Yes")
            else:
                print("No")
        elif a != 0 and b == 0:
            # No solution
            print("No")
        else:
            # T must have length a * |S| / b
            if a * b < 0:  # Different signs
                print("No")
            elif a * len(S) % b != 0:
                print("No")
            else:
                T_len = a * len(S) // b
                if can_match(S, T_len, X, Y):
                    print("Yes")
                else:
                    print("No")

def can_match(S, T_len, X, Y):
    if T_len == 0:
        # Special case: T is empty
        output_X = ''.join(S for c in X if c == '0')
        output_Y = ''.join(S for c in Y if c == '0')
        return output_X == output_Y
    
    # Build the output strings symbolically
    output_X = []
    for c in X:
        if c == '0':
            output_X.extend([('S', i) for i in range(len(S))])
        else:
            output_X.extend([('T', i) for i in range(T_len)])
    
    output_Y = []
    for c in Y:
        if c == '0':
            output_Y.extend([('S', i) for i in range(len(S))])
        else:
            output_Y.extend([('T', i) for i in range(T_len)])
    
    # Check if lengths match
    if len(output_X) != len(output_Y):
        return False
    
    # Use union-find to track equivalence classes of positions in T
    parent = list(range(T_len))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Track the required value for each position in T
    T_values = [None] * T_len
    
    # Process constraints
    for i in range(len(output_X)):
        source_X, pos_X = output_X[i]
        source_Y, pos_Y = output_Y[i]
        
        if source_X == 'S' and source_Y == 'S':
            if S[pos_X] != S[pos_Y]:
                return False
        elif source_X == 'S' and source_Y == 'T':
            if T_values[pos_Y] is not None and T_values[pos_Y] != S[pos_X]:
                return False
            T_values[pos_Y] = S[pos_X]
        elif source_X == 'T' and source_Y == 'S':
            if T_values[pos_X] is not None and T_values[pos_X] != S[pos_Y]:
                return False
            T_values[pos_X] = S[pos_Y]
        else:  # source_X == 'T' and source_Y == 'T'
            if pos_X != pos_Y:
                union(pos_X, pos_Y)
    
    # Check consistency within each equivalence class
    class_value = {}
    for i in range(T_len):
        root = find(i)
        if T_values[i] is not None:
            if root in class_value:
                if class_value[root] != T_values[i]:
                    return False
            else:
                class_value[root] = T_values[i]
    
    return True

solve()