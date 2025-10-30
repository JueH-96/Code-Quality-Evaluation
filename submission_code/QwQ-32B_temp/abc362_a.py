# Read the input values
r, g, b = map(int, input().split())
c = input().strip()

# Determine the minimum cost based on the forbidden color
if c == 'Red':
    print(min(g, b))
elif c == 'Green':
    print(min(r, b))
else:  # Blue case
    print(min(r, g))