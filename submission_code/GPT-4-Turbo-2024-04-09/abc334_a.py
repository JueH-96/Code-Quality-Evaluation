# Read input values
B, G = map(int, input().split())

# Determine which item is more expensive and print the corresponding output
if B > G:
    print("Bat")
else:
    print("Glove")