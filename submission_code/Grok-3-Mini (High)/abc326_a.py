# Read input from stdin
X, Y = map(int, input().split())

# Calculate the difference
diff = Y - X

# Check the condition for using stairs
if (diff > 0 and diff <= 2) or (diff < 0 and diff >= -3):
    print("Yes")
else:
    print("No")