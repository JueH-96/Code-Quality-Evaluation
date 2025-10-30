# Read the input
A, B = map(int, input().split())

# Calculate the difference
diff = abs(A - B)

# If A and B are the same, there's only one possible x
if diff == 0:
    print(1)
else:
    # There are two possible x values that can be in the middle of A and B
    # And diff - 1 possible x values that can be on either side of A and B
    print(2 + (diff - 1))