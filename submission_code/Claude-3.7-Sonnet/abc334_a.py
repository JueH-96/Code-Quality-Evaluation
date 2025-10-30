# Read the input
B, G = map(int, input().split())

# Determine which one is more expensive
if B > G:
    print("Bat")
else:
    print("Glove")