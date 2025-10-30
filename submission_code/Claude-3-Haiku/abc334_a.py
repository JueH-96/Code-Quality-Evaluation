# Read the input from stdin
b, g = map(int, input().split())

# Determine which item is more expensive
if b > g:
    print("Bat")
else:
    print("Glove")