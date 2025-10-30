def max_servings(N, Q, A, B):
    max_total = 0
    # Iterate over possible number of dish A servings
    # Since Q_i can be up to 1e6 and N up to 10, we need an efficient way
    # We can limit the number of dish A servings by the minimum Q_i / A_i where A_i > 0
    # Similarly for dish B
    # So, we can iterate over possible dish A servings up to the minimum Q_i / A_i where A_i > 0
    # and for each, compute the maximum possible dish B servings
    # To find the maximum possible dish A servings, we need to find the minimum Q_i / A_i where A_i > 0
    # Similarly for dish B
    # So, first, find the maximum possible dish A servings
    max_A = float('inf')
    for i in range(N):
        if A[i] > 0:
            max_A = min(max_A, Q[i] // A[i])
    # Now, iterate over possible dish A servings from 0 to max_A
    for a in range(max_A + 1):
        # Calculate remaining ingredients after making a servings of dish A
        remaining = [Q[i] - a * A[i] for i in range(N)]
        # Now, find the maximum possible dish B servings with the remaining ingredients
        max_B = float('inf')
        for i in range(N):
            if B[i] > 0:
                max_B = min(max_B, remaining[i] // B[i])
        # Update the total servings
        total = a + max_B
        if total > max_total:
            max_total = total
    return max_total

# Read input
N = int(input())
Q = list(map(int, input().split()))
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Compute and print the result
print(max_servings(N, Q, A, B))