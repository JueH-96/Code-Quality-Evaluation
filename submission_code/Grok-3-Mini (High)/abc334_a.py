# Read input values
B, G = map(int, input().split())

# Compare the costs and print the result
if B > G:
    print("Bat")
else:
    print("Glove")