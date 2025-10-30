# YOUR CODE HERE
def generate_triples(N):
    triples = []
    for x in range(N + 1):
        for y in range(N + 1):
            for z in range(N + 1):
                if x + y + z <= N:
                    triples.append((x, y, z))
    return triples

def main():
    import sys
    input = sys.stdin.read
    N = int(input().strip())
    triples = generate_triples(N)
    for triple in triples:
        print(f"{triple[0]} {triple[1]} {triple[2]}")

if __name__ == "__main__":
    main()