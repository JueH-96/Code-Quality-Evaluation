import sys

def solve():
    N, X = map(int, input().split())
    teeth = []
    for _ in range(N):
        U, D = map(int, input().split())
        teeth.append((U, D))

    # Sort the teeth by the difference between upper and lower lengths
    teeth.sort(key=lambda x: abs(x[0] - x[1]))

    total_cost = 0
    H = None
    for U, D in teeth:
        if H is None:
            H = U + D
        elif abs(U + D - H) <= X:
            continue
        else:
            # Grind the tooth with the larger length
            if U > D:
                total_cost += 1
                U -= 1
            else:
                total_cost += 1
                D -= 1
            H = U + D

    return total_cost

print(solve())