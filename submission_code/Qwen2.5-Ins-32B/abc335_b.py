# YOUR CODE HERE
def print_triples(N):
    for x in range(N + 1):
        for y in range(N - x + 1):
            for z in range(N - x - y + 1):
                print(x, y, z)

if __name__ == "__main__":
    N = int(input().strip())
    print_triples(N)