# Read the input values and split them into integers B and G
B, G = map(int, input().split())

# Compare B and G to determine which is more expensive
if B > G:
    print("Bat")
else:
    print("Glove")