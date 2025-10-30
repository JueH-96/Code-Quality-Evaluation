S = input()
A_index = S.find('A')
total_distance = 0

# Calculate the distance from 'A' to the right
right_distance = 0
for i in range(1, 26):
    if S[i] < S[i-1]:
        total_distance += right_distance
        right_distance = 0
    right_distance += 1

# Calculate the distance from 'A' to the left
left_distance = 0
for i in range(25, 0, -1):
    if S[i] > S[i-1]:
        total_distance += left_distance
        left_distance = 0
    left_distance += 1

# Add the remaining distance
total_distance += min(right_distance, left_distance)

print(total_distance)