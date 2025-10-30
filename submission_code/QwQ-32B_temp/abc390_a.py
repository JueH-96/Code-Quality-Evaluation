# Read the input as a list of integers
A = list(map(int, input().split()))
sorted_A = [1, 2, 3, 4, 5]

# Check if already sorted
if A == sorted_A:
    print("No")
else:
    possible = False
    # Check each adjacent pair
    for i in range(len(A) - 1):
        # Create a copy to avoid modifying the original
        new_A = A.copy()
        new_A[i], new_A[i+1] = new_A[i+1], new_A[i]
        if new_A == sorted_A:
            possible = True
            # Once found, can break early for efficiency
            break
    print("Yes" if possible else "No")