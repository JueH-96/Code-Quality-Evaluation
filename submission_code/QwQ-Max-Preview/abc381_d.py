n = int(input())
A = list(map(int, input().split()))

groups = []
current_group = []
prev_end = -2  # Tracks the end index of the previous pair

for i in range(n - 1):
    if A[i] == A[i + 1]:
        # Check if current pair is consecutive to the previous one
        if i == prev_end + 1:
            current_group.append(A[i])
        else:
            if current_group:
                groups.append(current_group)
            current_group = [A[i]]
        prev_end = i + 1  # Update the end of the current pair
    else:
        if current_group:
            groups.append(current_group)
            current_group = []
        prev_end = -2  # Reset if current pair is invalid

# Add the last group if it exists
if current_group:
    groups.append(current_group)

max_length = 0

# Process each group to find the maximum valid subarray length
for group in groups:
    elements = set()
    left = 0
    current_max = 0
    for right in range(len(group)):
        # Remove elements from the left until current element is not in the set
        while group[right] in elements:
            elements.remove(group[left])
            left += 1
        elements.add(group[right])
        current_max = max(current_max, right - left + 1)
    if current_max > max_length:
        max_length = current_max

print(max_length * 2)