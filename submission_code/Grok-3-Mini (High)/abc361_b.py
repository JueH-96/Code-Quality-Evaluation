import sys
# Read all input and split into a list of strings
data = sys.stdin.read().split()
# Convert to integers and unpack
a, b, c, d, e, f, g, h, i, j, k, l = map(int, data)

# Check for overlap in each dimension
x_overlap = max(a, g) < min(d, j)
y_overlap = max(b, h) < min(e, k)
z_overlap = max(c, i) < min(f, l)

# If all dimensions overlap, print Yes, else No
if x_overlap and y_overlap and z_overlap:
    print("Yes")
else:
    print("No")