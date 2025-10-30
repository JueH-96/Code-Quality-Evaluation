# Read inputs from standard input
R, G, B = map(int, input().split())
C = input().strip()

# Determine the minimum cost excluding the disliked color
if C == "Red":
    print(min(G, B))
elif C == "Green":
    print(min(R, B))
elif C == "Blue":
    print(min(R, G))