# YOUR CODE HERE
import sys

def solve():
    N, M = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))

    # Sort A in ascending order
    A.sort()

    # Check if it's possible to satisfy the condition
    if len([x for x in A if x >= max(B)]) < M:
        return -1

    # Create a list of (candy, price) tuples and sort by candy count
    boxes = sorted([(candy, price) for price, candy in enumerate(A, 1)], reverse=True)

    total_cost = 0
    for required_candy in sorted(B, reverse=True):
        for i, (candy, price) in enumerate(boxes):
            if candy >= required_candy:
                total_cost += A[price - 1]
                boxes.pop(i)
                break

    return total_cost

print(solve())