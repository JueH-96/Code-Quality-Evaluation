import sys

def solve():
    """
    Reads input, solves the problem, and prints the output.
    """
    try:
        # For faster I/O in Python
        input = sys.stdin.readline

        # Read problem dimensions
        N, M = map(int, input().split())

        # Read box prices/candy counts
        A = list(map(int, input().split()))

        # Read people's candy requirements
        B = list(map(int, input().split()))
    except (ValueError, IndexError):
        # Handles empty lines or malformed input
        return

    # According to problem constraints, M <= N.
    # If M were greater than N, it would be impossible.

    # --- The Greedy Algorithm ---
    # Sort both the boxes and requirements in ascending order.
    A.sort()
    B.sort()

    # Use a two-pointer approach to find the minimum cost assignment.
    # a_ptr points to the current box being considered.
    # b_ptr points to the current requirement being satisfied.
    a_ptr = 0
    b_ptr = 0
    total_cost = 0

    # Iterate until we have satisfied all M requirements or run out of boxes.
    while b_ptr < M and a_ptr < N:
        # Check if the current box can satisfy the current requirement.
        if A[a_ptr] >= B[b_ptr]:
            # Yes, it can. This is a valid match.
            # Add its cost to the total.
            total_cost += A[a_ptr]
            
            # Move on to the next requirement and the next box.
            b_ptr += 1
            a_ptr += 1
        else:
            # The current box is too small. Discard it and try the next one.
            a_ptr += 1

    # After the loop, check if all M requirements were met.
    # b_ptr will equal M if and only if we found a match for every requirement.
    if b_ptr == M:
        print(total_cost)
    else:
        # If b_ptr < M, we ran out of suitable boxes.
        print(-1)

solve()