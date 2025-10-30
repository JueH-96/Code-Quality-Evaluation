# Read the input from stdin
R = int(input())

# Calculate the minimum increase in rating required
if R < 100:
    increase = 100 - R
elif R < 200:
    increase = 200 - R
else:
    increase = 300 - R

# Print the result to stdout
print(increase)