# YOUR CODE HERE
N = int(input().strip())

# Water stations are located at every 5 km from 0 to 100
nearest_station = round(N / 5) * 5

# If the nearest station is behind Takahashi, check the next one
if nearest_station < N:
    nearest_station += 5

print(nearest_station)