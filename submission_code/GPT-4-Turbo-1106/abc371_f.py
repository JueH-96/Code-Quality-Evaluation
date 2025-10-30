import sys

# Read the number of persons and their initial positions
N = int(input())
X = list(map(int, input().split()))

# Read the number of tasks
Q = int(input())
tasks = [tuple(map(int, input().split())) for _ in range(Q)]

# Initialize the total movements
total_movements = 0

# Process each task
for T, G in tasks:
    person_index = T - 1
    current_position = X[person_index]
    movements = abs(G - current_position)

    # Update the positions of the persons that need to be moved
    if G < current_position:
        for i in range(person_index - 1, -1, -1):
            if X[i] >= G:
                movements += current_position - X[i]
                X[i] = current_position - 1
            current_position = X[i]
    else:
        for i in range(person_index + 1, N):
            if X[i] <= G:
                movements += X[i] - current_position
                X[i] = current_position + 1
            current_position = X[i]

    # Update the position of the person who completed the task
    X[person_index] = G

    # Add the movements for this task to the total movements
    total_movements += movements

# Output the total movements
print(total_movements)