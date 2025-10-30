# Read input values
N, T = map(int, input().split())
S = input().strip()
X = list(map(int, input().split()))

# Initialize the count of passing pairs
count = 0

# Create a list of tuples containing the position and direction of each ant
ants = [(X[i], 1 if S[i] == '1' else -1) for i in range(N)]

# Sort the ants by their initial positions
ants.sort()

# Iterate over each ant and count the number of pairs that will pass each other
for i in range(N):
    for j in range(i+1, N):
        # Calculate the distance between the ants
        distance = ants[j][0] - ants[i][0]
        # Calculate the relative speed at which they are approaching each other
        relative_speed = ants[i][1] + ants[j][1]
        # If the relative speed is 2, they are moving towards each other
        if relative_speed == 2:
            # If they can meet within time T, increment the count
            if distance / 2 <= T:
                count += 1
        # If the relative speed is 0, they are moving away from each other
        elif relative_speed == 0:
            # If they have already met, increment the count
            if distance / 2 <= 0.1:
                count += 1

# Print the result
print(count)