import sys

def read_ints():
    return list(map(int, sys.stdin.readline().split()))

def max_head_height(N, giants):
    # Sort the giants by the difference between their head and shoulder heights in descending order
    giants.sort(key=lambda x: x[1] - x[0], reverse=True)
    
    # Initialize the maximum height as the head height of the first giant
    max_height = giants[0][1]
    # Initialize the current shoulder height as the shoulder height of the first giant
    current_shoulder_height = giants[0][0]
    
    # Iterate over the remaining giants
    for i in range(1, N):
        # Add the shoulder height of the current giant to the current shoulder height
        current_shoulder_height += giants[i][0]
        # Update the maximum height if the current configuration results in a taller stack
        max_height = max(max_height, current_shoulder_height + giants[i][1])
    
    return max_height

# Read the number of giants
N = int(sys.stdin.readline())
# Read the shoulder and head heights of the giants
giants = [read_ints() for _ in range(N)]

# Calculate and print the maximum possible height of the head of the topmost giant
print(max_head_height(N, giants))