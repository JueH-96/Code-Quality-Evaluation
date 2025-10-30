# Read the input
N = int(input())
keys = []
for _ in range(N):
    a, s = input().split()
    keys.append((int(a), s))

# Initialize the hands' positions
left_hand = 1
right_hand = 1
fatigue = 0

# Process the key presses
for a, s in keys:
    if s == 'L':
        fatigue += abs(a - left_hand)
        left_hand = a
    else:
        fatigue += abs(a - right_hand)
        right_hand = a

# Print the minimum fatigue level
print(fatigue)