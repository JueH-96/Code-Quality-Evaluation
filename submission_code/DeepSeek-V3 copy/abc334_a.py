# Read the input values for B and G
B, G = map(int, input().split())

# Determine which item is more expensive
if B > G:
    print("Bat")
else:
    print("Glove")