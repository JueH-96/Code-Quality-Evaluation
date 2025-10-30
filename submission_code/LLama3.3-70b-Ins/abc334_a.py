# Read the input from stdin
B, G = map(int, input().split())

# Determine which one is more expensive
if B > G:
    print("Bat")
else:
    print("Glove")