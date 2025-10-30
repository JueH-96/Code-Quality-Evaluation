# YOUR CODE HERE
def calculate_cost(N):
    cost = 0
    while N > 1:
        cost += N
        N = (N + 1) // 2
    return cost

N = int(input())
print(calculate_cost(N))