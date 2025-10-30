N = int(input())
giants = []
for i in range(N):
    a, b = map(int, input().split())
    giants.append((a, b, b-a))

# Sort by difference between head and shoulder height in descending order
giants.sort(key=lambda x: x[2], reverse=True)

# Calculate total height by stacking giants
total_height = 0
for i in range(N):
    total_height += giants[i][0]  # Add shoulder height

# Add head-shoulder difference of last giant
total_height += giants[-1][2]

print(total_height)