N, M = map(int, input().split())
S = input()

def can_satisfy(logo_shirts):
    # Total shirts = plain + logo shirts
    total_shirts = M + logo_shirts
    
    # Track available clean shirts
    clean_plain = M
    clean_logo = logo_shirts
    
    # Track dirty shirts
    dirty_plain = 0
    dirty_logo = 0
    
    for day in S:
        if day == '0':  # No plans - wash all shirts
            clean_plain += dirty_plain
            clean_logo += dirty_logo
            dirty_plain = 0
            dirty_logo = 0
            continue
            
        if day == '1':  # Meal - need any shirt
            if clean_plain > 0:  # Prefer using plain shirt if available
                clean_plain -= 1
                dirty_plain += 1
            elif clean_logo > 0:  # Use logo shirt if no plain available
                clean_logo -= 1
                dirty_logo += 1
            else:
                return False
                
        if day == '2':  # Event - need logo shirt
            if clean_logo > 0:
                clean_logo -= 1
                dirty_logo += 1
            else:
                return False
                
    return True

# Binary search for minimum number of logo shirts needed
left = 0
right = N

result = -1
while left <= right:
    mid = (left + right) // 2
    if can_satisfy(mid):
        result = mid
        right = mid - 1
    else:
        left = mid + 1

print(result)