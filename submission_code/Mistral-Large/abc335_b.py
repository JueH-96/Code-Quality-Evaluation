import sys

def find_triples(N):
    for x in range(N + 1):
        for y in range(N + 1 - x):
            for z in range(N + 1 - x - y):
                print(x, y, z)

if __name__ == "__main__":
    input = sys.stdin.read()
    N = int(input.strip())
    find_triples(N)