# YOUR CODE HERE

N, K = map(int, input().split())
functions = [tuple(map(int, input().split())) for _ in range(N)]

def calculate_function(p, x):
    for i in p:
        x = functions[i-1][0]*x + functions[i-1][1]
    return x

max_value = float('-inf')
for p in permutations(range(1, N+1), K):
    max_value = max(max_value, calculate_function(p, 1))

print(max_value)