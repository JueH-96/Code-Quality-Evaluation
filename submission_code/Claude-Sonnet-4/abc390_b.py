# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

# For a sequence of length 2, it's always a geometric progression
if n == 2:
    print("Yes")
else:
    # Calculate the first ratio
    ratio = a[1] / a[0]
    
    # Check if all consecutive ratios are equal to the first ratio
    is_geometric = True
    for i in range(2, n):
        if a[i-1] == 0:  # Avoid division by zero
            is_geometric = False
            break
        current_ratio = a[i] / a[i-1]
        if current_ratio != ratio:
            is_geometric = False
            break
    
    if is_geometric:
        print("Yes")
    else:
        print("No")