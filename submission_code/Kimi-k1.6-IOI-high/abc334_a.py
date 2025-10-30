# Read the input values
B, G = map(int, input().split())

# Compare the prices and print the result
if B > G:
    print("Bat")
else:
    print("Glove")