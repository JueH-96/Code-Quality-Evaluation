# Read the input
R, G, B = map(int, input().split())
C = input()

# Determine the minimum amount of money needed to buy one pen
if C == 'Red':
    print(min(G, B))
elif C == 'Green':
    print(min(R, B))
else:  # C == 'Blue'
    print(min(R, G))