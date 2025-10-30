# Read the input from stdin
N = int(input())

# Generate and print all triples in lexicographical order
for x in range(N + 1):
    for y in range(N - x + 1):
        for z in range(N - x - y + 1):
            print(x, y, z)