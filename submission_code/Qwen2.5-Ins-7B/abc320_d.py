# YOUR CODE HERE
from collections import defaultdict
from typing import List, Tuple

def solve(N: int, M: int, queries: List[Tuple[int, int, int, int]]) -> List[str]:
    # Initialize the positions
    positions = defaultdict(lambda: (None, None))
    positions[1] = (0, 0)
    
    # Process each query
    for A, B, X, Y in queries:
        if positions[A][0] is None or positions[A][1] is None:
            # If the position of A is not known, we cannot determine the position of B
            positions[B] = (None, None)
        else:
            # Calculate the position of B relative to A
            x_a, y_a = positions[A]
            x_b, y_b = x_a + X, y_a + Y
            if positions[B][0] is None or positions[B][1] is None:
                # If the position of B is not known, update it
                positions[B] = (x_b, y_b)
            else:
                # If the position of B is known, check for consistency
                x_b_known, y_b_known = positions[B]
                if (x_b_known, y_b_known) != (x_b, y_b):
                    return [f"undecidable"] * N
    
    # Determine the final positions
    final_positions = {}
    for i in range(1, N + 1):
        if positions[i][0] is None or positions[i][1] is None:
            final_positions[i] = "undecidable"
        else:
            x, y = positions[i]
            final_positions[i] = (x, y)
    
    # Format the output
    return [f"{x} {y}" if x is not None and y is not None else "undecidable" for i, (x, y) in final_positions.items()]

if __name__ == "__main__":
    N, M = map(int, input().split())
    queries = [tuple(map(int, input().split())) for _ in range(M)]
    results = solve(N, M, queries)
    for result in results:
        print(result)