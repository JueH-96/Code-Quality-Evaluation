# Read the input from stdin
B, G = map(int, input().split())

# Determine which item is more expensive and print the result
if B > G:
    print("Bat")
else:
    print("Glove")