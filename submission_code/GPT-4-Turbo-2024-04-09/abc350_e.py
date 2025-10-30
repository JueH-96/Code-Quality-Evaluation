import sys
import functools

input = sys.stdin.read
@functools.lru_cache(None)
def min_cost(n):
    if n == 0:
        return 0.0
    
    # Cost using operation A
    cost_A = X + min_cost(n // A)
    
    # Expected cost using operation B (rolling the die)
    expected_cost_B = 0.0
    for b in range(1, 7):
        expected_cost_B += min_cost(n // b)
    expected_cost_B = Y + (expected_cost_B / 6)
    
    # Return the minimum of both operations
    return min(cost_A, expected_cost_B)

def main():
    global N, A, X, Y
    N, A, X, Y = map(int, input().split())
    result = min_cost(N)
    print(f"{result:.12f}")

if __name__ == "__main__":
    main()