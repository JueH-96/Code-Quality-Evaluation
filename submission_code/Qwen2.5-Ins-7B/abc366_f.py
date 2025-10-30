# NUMBER_OF_FUNCTIONS NUMBER_OF_NESTINGS
N, K = map(int, input().split())
# COEFFICIENTS AND CONSTANTS
functions = [tuple(map(int, input().split())) for _ in range(N)]

# DETERMINE THE FUNCTION THAT YIELDS THE HIGHEST VALUE FOR f(x) = A*x + B
max_function = max(functions, key=lambda f: f[0]*N + f[1])

# CALCULATE THE MAXIMUM POSSIBLE VALUE FOR THE COMPOSITION OF K FUNCTIONS
current_value = 1
for _ in range(K):
    current_value = max_function[0] * current_value + max_function[1]

print(current_value)