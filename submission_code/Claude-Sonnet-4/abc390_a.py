# YOUR CODE HERE
A = list(map(int, input().split()))
target = [1, 2, 3, 4, 5]

# If already sorted, we can't perform exactly one operation to sort it
if A == target:
    print("No")
else:
    # Try swapping each pair of adjacent elements
    can_sort = False
    
    for i in range(4):  # positions 0,1,2,3 (can swap with next element)
        # Create a copy and swap adjacent elements
        temp = A[:]
        temp[i], temp[i+1] = temp[i+1], temp[i]
        
        # Check if this results in sorted array
        if temp == target:
            can_sort = True
            break
    
    if can_sort:
        print("Yes")
    else:
        print("No")