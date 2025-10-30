# YOUR CODE HERE
N = int(input())

# Initialize the nearest_distance to a large number
nearest_distance = 10000

# Iterate over the possible positions of the water stations
for i in range(0, 101, 5):
    # Calculate the distance between Takahashi and the current water station
    distance = abs(N - i)
    # If the distance is smaller than the current nearest_distance, update nearest_distance
    if distance < nearest_distance:
        nearest_distance = distance

print(nearest_distance)