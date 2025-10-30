# Reading the input values for B and G
B, G = map(int, input().split())

# Comparing the values of B and G to determine which one is more expensive
if B > G:
    print("Bat")
else:
    print("Glove")