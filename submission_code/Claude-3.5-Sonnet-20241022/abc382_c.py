# Read input
N, M = map(int, input().split())
A = list(map(int, input().split()))  # Gourmet levels
B = list(map(int, input().split()))  # Sushi deliciousness

# Process each sushi
for sushi_deliciousness in B:
    # For each sushi, check each person in order
    eaten = False
    for i in range(N):
        # If person's gourmet level is less than or equal to sushi deliciousness
        if A[i] <= sushi_deliciousness:
            print(i + 1)  # Print person number (1-based indexing)
            eaten = True
            break
    
    # If nobody ate the sushi
    if not eaten:
        print(-1)