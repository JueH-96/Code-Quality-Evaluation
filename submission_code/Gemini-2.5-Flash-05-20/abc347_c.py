import sys

def solve():
    N, A, B = map(int, sys.stdin.readline().split())
    D = list(map(int, sys.stdin.readline().split()))

    W = A + B

    # Calculate D_i % W for all plans
    D_mod_W = [d % W for d in D]

    # Sort the D_i % W values. This is crucial for checking contiguous spans.
    D_mod_W.sort()

    # Create an extended list to handle circularity.
    # This effectively doubles the sorted list and shifts the second half by W.
    # [v_0, v_1, ..., v_{N-1}, v_0+W, v_1+W, ..., v_{N-1}+W]
    # This allows us to consider all possible contiguous N-element windows that
    # might wrap around the end of the W-day week.
    extended_D_mod_W = D_mod_W + [d + W for d in D_mod_W]

    # Initialize min_required_length with a value larger than any possible span.
    # The maximum possible span is W (e.g., if all plans are at D_i % W = 0 and A=1).
    # If N=1, extended_D_mod_W[0+1-1] - extended_D_mod_W[0] + 1 = 0 + 1 = 1.
    # W is a safe upper bound for initializing.
    min_required_length = W + 1 

    # Iterate through all possible starting points (v_i) in the sorted original values.
    # For each v_i, consider the N plans that would span from v_i to v_{i+N-1} (in the extended list).
    for i in range(N):
        # Calculate the length of the interval required to cover these N points.
        # This is (last_point - first_point + 1).
        # extended_D_mod_W[i + N - 1] is the (N-1)-th point after extended_D_mod_W[i].
        current_span_length = extended_D_mod_W[i + N - 1] - extended_D_mod_W[i] + 1
        
        # Update min_required_length if the current span is smaller.
        min_required_length = min(min_required_length, current_span_length)

    # If the minimum length required to cover all plans is less than or equal to A
    # (the number of holiday days), then it's possible to shift the week's starting
    # day such that all plans fall on holidays.
    if A >= min_required_length:
        print("Yes")
    else:
        print("No")

solve()