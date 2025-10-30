import sys
from itertools import permutations

def max_function_value(N, K, functions):
    max_value = 0

    # Generate all permutations of K distinct numbers from 1 to N
    for p in permutations(range(1, N + 1), K):
        value = 1
        for index in p:
            A, B = functions[index - 1]
            value = A * value + B
        max_value = max(max_value, value)

    return max_value

def main():
    input = sys.stdin.read
    data = input().split()

    index = 0
    N = int(data[index])
    index += 1
    K = int(data[index])
    index += 1

    functions = []
    for _ in range(N):
        A = int(data[index])
        index += 1
        B = int(data[index])
        index += 1
        functions.append((A, B))

    result = max_function_value(N, K, functions)
    print(result)

if __name__ == "__main__":
    main()