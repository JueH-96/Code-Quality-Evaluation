# Read the input
N = int(input())
times_and_volumes = []
for _ in range(N):
    t, v = map(int, input().split())
    times_and_volumes.append((t, v))

# Initialize the water level
water_level = 0

# Process the additions of water
for t, v in times_and_volumes:
    # Decrease the water level by t - previous_time
    water_level = max(0, water_level - (t - times_and_volumes[0][0]))
    # Add the new water
    water_level += v

# Print the final water level
print(water_level)