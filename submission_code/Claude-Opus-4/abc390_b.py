# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

# A sequence with 2 elements is always a geometric progression
if n == 2:
    print("Yes")
else:
    # Check if all ratios are equal using cross multiplication
    # For a geometric progression: a[i+1]/a[i] = a[i]/a[i-1]
    # Which means: a[i+1] * a[i-1] = a[i] * a[i]
    is_geometric = True
    
    for i in range(1, n - 1):
        # Check if a[i+1]/a[i] == a[i]/a[i-1]
        # Using cross multiplication: a[i+1] * a[i-1] == a[i] * a[i]
        if a[i + 1] * a[i - 1] != a[i] * a[i]:
            is_geometric = False
            break
    
    if is_geometric:
        print("Yes")
    else:
        print("No")