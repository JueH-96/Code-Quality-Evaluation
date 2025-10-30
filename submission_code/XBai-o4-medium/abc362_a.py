# Read the input values
R, G, B = map(int, input().split())
C = input().strip()

# Determine the minimum cost based on the disliked color
if C == 'Red':
    print(min(G, B))
elif C == 'Green':
    print(min(R, B))
else:  # Blue
    print(min(R, G))