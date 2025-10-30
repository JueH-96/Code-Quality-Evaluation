# Read input from stdin
x, y = map(int, input().split())

# Calculate the difference in floors
diff = y - x

# Check if moving up (diff > 0) and up 2 or less (diff <= 2)
# OR if moving down (diff < 0) and down 3 or less (abs(diff) <= 3)
if (diff > 0 and diff <= 2) or (diff < 0 and abs(diff) <= 3):
    print("Yes")
else:
    print("No")