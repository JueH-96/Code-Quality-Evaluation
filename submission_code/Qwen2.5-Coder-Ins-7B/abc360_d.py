# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
T = int(data[1])
S = data[2]
X = list(map(int, data[3:]))

# Separate ants into two lists based on their direction
left_ants = [(X[i], i) for i in range(N) if S[i] == '0']
right_ants = [(X[i], i) for i in range(N) if S[i] == '1']

# Sort ants by their position
left_ants.sort()
right_ants.sort()

# Initialize variables to count the number of passing pairs
count = 0
right_index = 0

# Iterate over left-moving ants
for left_pos, left_index in left_ants:
    # Move the right-moving ant pointer to the first ant that can pass the left-moving ant
    while right_index < len(right_ants) and right_ants[right_index][0] < left_pos:
        right_index += 1
    
    # Count the number of passing pairs
    count += len(right_ants) - right_index

# Output the result
print(count)