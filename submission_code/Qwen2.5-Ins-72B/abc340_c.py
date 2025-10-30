# YOUR CODE HERE
def solve(N):
    total_cost = 0
    while N >= 2:
        total_cost += N
        N = (N + 1) // 2
    print(total_cost)

N = int(input())
solve(N)