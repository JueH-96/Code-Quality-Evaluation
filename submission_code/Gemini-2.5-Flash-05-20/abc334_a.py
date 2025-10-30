# Read the input line
line = input()

# Split the line into two parts and convert them to integers
B, G = map(int, line.split())

# Compare the costs and print the result
if B > G:
    print("Bat")
else: # G > B, because B != G is guaranteed
    print("Glove")