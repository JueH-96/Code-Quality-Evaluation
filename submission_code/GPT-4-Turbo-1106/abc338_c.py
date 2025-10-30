def max_servings(N, Q, A, B):
    # Initialize the maximum servings for each dish to a large number
    max_servings_A = max_servings_B = float('inf')
    
    # Calculate the maximum number of servings for dish A
    for i in range(N):
        if A[i] > 0:  # Avoid division by zero
            max_servings_A = min(max_servings_A, Q[i] // A[i])
    
    # Calculate the maximum number of servings for dish B
    for i in range(N):
        if B[i] > 0:  # Avoid division by zero
            max_servings_B = min(max_servings_B, Q[i] // B[i])
    
    # Calculate the maximum total number of servings
    max_total_servings = 0
    for i in range(max_servings_A + 1):
        # Calculate the remaining ingredients after making i servings of dish A
        remaining_ingredients = [Q[j] - A[j] * i for j in range(N)]
        
        # Calculate the maximum number of servings of dish B with the remaining ingredients
        max_servings_B_with_remaining = float('inf')
        for j in range(N):
            if B[j] > 0:  # Avoid division by zero
                max_servings_B_with_remaining = min(max_servings_B_with_remaining, remaining_ingredients[j] // B[j])
        
        # Update the maximum total number of servings
        max_total_servings = max(max_total_servings, i + max_servings_B_with_remaining)
    
    return max_total_servings

# Read input from stdin
N = int(input().strip())
Q = list(map(int, input().strip().split()))
A = list(map(int, input().strip().split()))
B = list(map(int, input().strip().split()))

# Calculate and print the answer
print(max_servings(N, Q, A, B))