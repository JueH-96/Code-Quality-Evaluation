W, B = map(int, input().split())

# The pattern of the piano keyboard
pattern = "wbwbwwbwbwbw"

# Count the number of white and black keys in the pattern
white_keys = pattern.count('w')
black_keys = pattern.count('b')

# Check if there is a continuous segment with W white keys and B black keys
if (W == 0 or W <= white_keys) and (B == 0 or B <= black_keys):
    print("Yes")
else:
    print("No")