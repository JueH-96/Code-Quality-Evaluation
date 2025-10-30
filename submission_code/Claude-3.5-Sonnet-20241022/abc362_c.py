def solve():
    N = int(input())
    pairs = []
    for _ in range(N):
        l, r = map(int, input().split())
        pairs.append((l, r))
    
    # Calculate sum ranges
    min_sum = sum(p[0] for p in pairs)
    max_sum = sum(p[1] for p in pairs)
    
    # If 0 is not in possible range of sums, no solution exists
    if min_sum > 0 or max_sum < 0:
        print("No")
        return
        
    # Try to construct a solution
    result = []
    target = 0
    curr_sum = 0
    
    # Take minimum values for first N-1 elements
    for i in range(N-1):
        result.append(pairs[i][0])
        curr_sum += pairs[i][0]
    
    # Last element needs to make total sum 0
    needed = -curr_sum
    if pairs[N-1][0] <= needed <= pairs[N-1][1]:
        result.append(needed)
        print("Yes")
        print(*result)
    else:
        # If last element can't make sum 0, try adjusting previous elements
        can_adjust = False
        for i in range(N-1):
            curr_sum -= result[i]  # Remove current element from sum
            space_available = pairs[i][1] - pairs[i][0]  # How much we can increase this element
            needed_adjustment = -curr_sum
            
            if pairs[i][0] <= pairs[i][0] + needed_adjustment <= pairs[i][1]:
                # We can adjust this element to make sum 0
                result[i] = pairs[i][0] + needed_adjustment
                can_adjust = True
                break
            
            # Couldn't adjust this element, put back original value
            result[i] = pairs[i][0]
            curr_sum += result[i]
        
        if can_adjust:
            print("Yes")
            print(*result)
        else:
            print("No")

solve()