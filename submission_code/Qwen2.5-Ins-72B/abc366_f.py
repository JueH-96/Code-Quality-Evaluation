# YOUR CODE HERE
import itertools

def max_linear_function_value(N, K, functions):
    def apply_function(x, A, B):
        return A * x + B

    def compute_value(sequence):
        x = 1
        for i in sequence:
            x = apply_function(x, functions[i-1][0], functions[i-1][1])
        return x

    max_value = 0
    for sequence in itertools.permutations(range(1, N+1), K):
        max_value = max(max_value, compute_value(sequence))
    
    return max_value

if __name__ == "__main__":
    N, K = map(int, input().split())
    functions = [tuple(map(int, input().split())) for _ in range(N)]
    print(max_linear_function_value(N, K, functions))