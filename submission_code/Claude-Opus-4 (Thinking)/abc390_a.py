# YOUR CODE HERE
A = list(map(int, input().split()))

# Check if array is already sorted
if A == [1, 2, 3, 4, 5]:
    print("No")
else:
    found = False
    for i in range(4):  # 0 to 3, since we're swapping i with i+1
        # Make a copy and swap
        B = A.copy()
        B[i], B[i+1] = B[i+1], B[i]
        
        # Check if sorted
        if B == [1, 2, 3, 4, 5]:
            found = True
            break
    
    if found:
        print("Yes")
    else:
        print("No")