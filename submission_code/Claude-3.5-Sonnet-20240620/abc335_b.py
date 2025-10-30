# YOUR CODE HERE
def print_triples(N):
    for x in range(N + 1):
        for y in range(N + 1 - x):
            for z in range(N + 1 - x - y):
                print(f"{x} {y} {z}")

N = int(input())
print_triples(N)