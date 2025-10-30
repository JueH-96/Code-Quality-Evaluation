# Read the input values and split them into integers B and G
B, G = map(int, input().split())

# Compare the two values and print the appropriate result
if B > G:
    print("Bat")
else:
    print("Glove")