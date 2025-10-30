# YOUR CODE HERE
A = list(map(int, input().split()))

# Check if already sorted
if A == [1, 2, 3, 4, 5]:
    print("No")
else:
    # Try swapping each adjacent pair
    found = False
    for i in range(4):
        # Make a copy and swap adjacent elements
        temp = A.copy()
        temp[i], temp[i+1] = temp[i+1], temp[i]
        
        # Check if sorted
        if temp == [1, 2, 3, 4, 5]:
            found = True
            break
    
    if found:
        print("Yes")
    else:
        print("No")