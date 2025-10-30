# Read the input
N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Initialize a list to store the person who eats each sushi
eaters = [-1] * M

# Iterate through the sushi
for j in range(M):
    # Iterate through the people
    for i in range(N):
        # If the sushi's deliciousness is greater than or equal to the person's gourmet level
        if B[j] >= A[i]:
            # The person eats the sushi
            eaters[j] = i + 1
            # The sushi will not pass in front of the remaining people
            break

# Print the output
for e in eaters:
    print(e)