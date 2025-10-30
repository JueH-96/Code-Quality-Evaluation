import sys
import itertools

input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])

functions = []
index = 2
for _ in range(N):
    A = int(data[index])
    B = int(data[index + 1])
    functions.append((A, B))
    index += 2

def evaluate_function_chain(functions, indices, initial_value):
    value = initial_value
    for index in indices:
        A, B = functions[index]
        value = A * value + B
    return value

max_value = float('-inf')
for indices in itertools.permutations(range(N), K):
    result = evaluate_function_chain(functions, indices, 1)
    if result > max_value:
        max_value = result

print(max_value)