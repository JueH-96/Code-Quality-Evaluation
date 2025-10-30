def solve():
    n = int(input())
    giants = []
    for _ in range(n):
        a, b = map(int, input().split())
        giants.append((a, b))

    import itertools
    
    max_height = 0
    for permutation in itertools.permutations(range(n)):
        current_height = 0
        
        for i in range(n):
            giant_index = permutation[i]
            a, b = giants[giant_index]
            
            if i == 0:
                current_height = b
            else:
                current_height += a
                current_height = current_height - giants[permutation[i-1]][0] + giants[permutation[i-1]][1]
                current_height = current_height - a + b
                
        
        
        current_height = 0
        
        for i in range(n):
            giant_index = permutation[i]
            a, b = giants[giant_index]
            
            if i == 0:
                current_height = b
            else:
                current_height += a
        
        max_height = max(max_height, current_height)
    
    
    max_height = 0
    for permutation in itertools.permutations(range(n)):
        current_height = 0
        
        for i in range(n):
            giant_index = permutation[i]
            a, b = giants[giant_index]
            
            if i == 0:
                current_height = b
            else:
                current_height += a
        
        max_height = max(max_height, current_height)

    print(max_height)

solve()