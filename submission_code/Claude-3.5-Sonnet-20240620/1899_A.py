# YOUR CODE HERE
def solve(n):
    for move in range(10):
        if move % 2 == 0:  # Vanya's turn
            if (n + 1) % 3 == 0 or (n - 1) % 3 == 0:
                return "First"
            n += 1 if (n + 1) % 3 < (n - 1) % 3 else -1
        else:  # Vova's turn
            n += 1 if (n + 1) % 3 > (n - 1) % 3 else -1
    return "Second"

t = int(input())
for _ in range(t):
    n = int(input())
    print(solve(n))