X, Y, Z = map(int, input().split())
S = input().strip()

prev_0 = 0
prev_1 = float('inf')

for c in S:
    new_0 = float('inf')
    new_1 = float('inf')
    
    # Transitions from prev_0
    if prev_0 != float('inf'):
        for m in [0, 1]:
            s_new = 0 ^ m
            if s_new == 0:
                cost = X if c == 'a' else Y
            else:
                cost = X if c == 'A' else Y
            total = prev_0 + m * Z + cost
            if s_new == 0:
                new_0 = min(new_0, total)
            else:
                new_1 = min(new_1, total)
    
    # Transitions from prev_1
    if prev_1 != float('inf'):
        for m in [0, 1]:
            s_new = 1 ^ m
            if s_new == 0:
                cost = X if c == 'a' else Y
            else:
                cost = X if c == 'A' else Y
            total = prev_1 + m * Z + cost
            if s_new == 0:
                new_0 = min(new_0, total)
            else:
                new_1 = min(new_1, total)
    
    prev_0, prev_1 = new_0, new_1

print(min(prev_0, prev_1))