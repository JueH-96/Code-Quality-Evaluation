# YOUR CODE HERE
import sys
from itertools import permutations

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    K = int(input[1])
    functions = []
    index = 2
    for _ in range(N):
        A = int(input[index])
        B = int(input[index + 1])
        functions.append((A, B))
        index += 2

    max_value = float('-inf')
    for perm in permutations(range(N), K):
        x = 1
        for i in perm:
            A, B = functions[i]
            x = A * x + B
        max_value = max(max_value, x)

    print(max_value)

if __name__ == "__main__":
    main()