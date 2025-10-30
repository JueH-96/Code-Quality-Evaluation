def minimize_slimes():
    n = int(input())
    slime_count = {}
    
    # Initialize the dictionary with the initial slimes
    for _ in range(n):
        size, count = map(int, input().split())
        slime_count[size] = count
    
    # Keep merging as long as there are pairs of slimes of the same size
    change = True
    while change:
        change = False
        new_slime_count = {}
        
        for size, count in slime_count.items():
            pairs = count // 2
            remainder = count % 2
            
            if pairs > 0:
                change = True
                if 2 * size in new_slime_count:
                    new_slime_count[2 * size] += pairs
                else:
                    new_slime_count[2 * size] = pairs
            
            if remainder > 0:
                if size in new_slime_count:
                    new_slime_count[size] += remainder
                else:
                    new_slime_count[size] = remainder
        
        slime_count = new_slime_count
    
    # Count the total number of slimes
    total_slimes = sum(slime_count.values())
    
    return total_slimes

print(minimize_slimes())