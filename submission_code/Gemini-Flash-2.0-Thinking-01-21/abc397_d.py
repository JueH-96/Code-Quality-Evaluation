import sys
import math

# Function to check if a large integer Z is a perfect square
# Returns True and the integer square root if it is, False and -1 otherwise.
def is_perfect_square(Z):
    # A perfect square cannot be negative.
    if Z < 0:
        return False, -1
    # 0 is a perfect square (0 * 0 = 0).
    if Z == 0:
        return True, 0
    
    # Estimate the integer square root using floating-point arithmetic.
    # Use float() to ensure Z is treated as a float for the power operation.
    # For very large integers, float precision might cause the estimate s_approx
    # to be slightly off from the true floor(sqrt(Z)).
    # The integer square root s must satisfy s*s <= Z.
    # The s_approx obtained might be floor(sqrt(Z)), floor(sqrt(Z))-1, or floor(sqrt(Z))+1.
    s_approx = int(float(Z)**0.5)
    
    # Check a small range of integers around the approximation.
    # A range of size 5 (s_approx - 2 to s_approx + 2) should be sufficient
    # to cover potential floating-point inaccuracies for numbers within the given constraints.
    # Use max(0, ...) to ensure we only check non-negative integers as square roots.
    for s in range(max(0, s_approx - 2), s_approx + 3):
        # Check if the square of the candidate integer equals the original number Z.
        if s * s == Z:
            return True, s # Found the integer square root
    
    # If no integer in the checked range squares to Z, Z is not a perfect square.
    return False, -1

# Main function to solve the problem.
def solve():
    # Read the input integer N from standard input.
    # N is a positive integer up to 10^18.
    N = int(sys.stdin.readline())

    # We are searching for positive integers x, y such that x^3 - y^3 = N.
    # Since N is positive and x, y are positive integers, we must have x^3 > y^3, which implies x > y.
    # Also, since y is a positive integer, the smallest possible value for y is 1. Thus y >= 1.
    # Since x > y and y >= 1, the smallest possible value for x is 2 (when y=1). Thus x >= 2.

    # Let k be the difference between x and y: k = x - y.
    # Since x > y and x, y are integers, k must be a positive integer (k >= 1).
    # Substituting x = y + k into the equation x^3 - y^3 = N:
    # (y + k)^3 - y^3 = N
    # Expanding (y + k)^3 gives y^3 + 3y^2k + 3yk^2 + k^3.
    # So the equation becomes (y^3 + 3y^2k + 3yk^2 + k^3) - y^3 = N
    # 3ky^2 + 3k^2y + k^3 = N

    # This is a quadratic equation in y of the form Ay^2 + By + C = 0,
    # where A = 3k, B = 3k^2, and C = k^3 - N.
    # We are looking for positive integer solutions for y.

    # Using the quadratic formula, the solutions for y are:
    # y = [-B +/- sqrt(B^2 - 4AC)] / (2A)
    # y = [-3k^2 +/- sqrt((3k^2)^2 - 4(3k)(k^3 - N))] / (2 * 3k)
    # y = [-3k^2 +/- sqrt(9k^4 - 12k^4 + 12kN)] / (6k)

    # Let Delta = 12kN - 3k^4 = 3k(4N - k^3) be the discriminant.
    # For real solutions for y, Delta must be non-negative (Delta >= 0).
    # 3k(4N - k^3) >= 0. Since k is a positive integer (k >= 1), k > 0.
    # So, 4N - k^3 >= 0, which means k^3 <= 4N.
    # This gives an upper bound for k: k <= (4N)^(1/3).

    # Since y must be a positive integer (y >= 1), we must choose the '+' sign in the quadratic formula:
    # y = (-3k^2 + sqrt(Delta)) / (6k).
    # For y >= 1, we need (-3k^2 + sqrt(Delta)) / (6k) >= 1.
    # Since 6k > 0 for k >= 1, this inequality is equivalent to:
    # -3k^2 + sqrt(Delta) >= 6k
    # sqrt(Delta) >= 3k^2 + 6k.
    # Since k >= 1, both sides are non-negative (3k^2 + 6k > 0). We can square both sides:
    # Delta >= (3k^2 + 6k)^2 = (3k(k + 2))^2 = 9k^2(k^2 + 4k + 4) = 9k^4 + 36k^3 + 36k^2.
    # Substitute Delta = 12kN - 3k^4 back into the inequality:
    # 12kN - 3k^4 >= 9k^4 + 36k^3 + 36k^2.
    # 12kN >= 12k^4 + 36k^3 + 36k^2.
    # Divide by 12k (since k >= 1, 12k > 0):
    # N >= k^3 + 3k^2 + 3k.
    # This inequality N >= k^3 + 3k^2 + 3k gives a tighter upper bound on k
    # for which a positive integer solution y >= 1 might exist.
    # If k^3 + 3k^2 + 3k > N, then for this k and any larger k, no positive integer y >= 1 satisfies the equation.

    # We can iterate through possible integer values of k starting from 1.
    k = 1
    while True:
        # Calculate k^2 and k^3 for the current k. Python handles large integers automatically.
        k2 = k * k
        k3 = k * k2

        # Calculate the value k^3 + 3k^2 + 3k to check the loop termination condition.
        # This value is the minimum N for a solution with difference k and y >= 1.
        k_cubic_poly = k3 + 3 * k2 + 3 * k

        # Check if the loop termination condition is met.
        # If k^3 + 3k^2 + 3k > N, we can stop searching for k.
        if k_cubic_poly > N:
            break # No solution found for this k or larger k with y >= 1.

        # If k_cubic_poly <= N, a positive integer solution y >= 1 might exist for this k.
        # Calculate the discriminant Delta = 3k(4N - k^3).
        # We know that if k_cubic_poly <= N, then k^3 <= N <= 4N (since N >= 1).
        # So, 4N - k^3 >= 0, which implies Delta = 3k(4N - k^3) >= 0.
        term_4N_minus_k3 = 4 * N - k3
        delta = 3 * k * term_4N_minus_k3

        # For y to be rational, Delta must be a perfect square.
        # Check if Delta is a perfect square and get its integer square root.
        is_sq, sqrt_delta = is_perfect_square(delta)

        # If Delta is not a perfect square, there is no rational solution for y for this k.
        # Move to the next value of k.
        if not is_sq:
            k += 1
            continue

        # If Delta is a perfect square, sqrt_delta is its integer square root.
        # The potential integer solution for y is y = (-3k^2 + sqrt(Delta)) / (6k).
        # Calculate the numerator and the denominator.
        numerator = -3 * k2 + sqrt_delta
        denominator = 6 * k

        # For y to be a positive integer (y >= 1), the numerator must satisfy:
        # 1. numerator >= 0 (for y >= 0)
        # 2. numerator is divisible by the denominator (for y to be an integer)
        # 3. Resulting quotient y = numerator / denominator must be > 0 (y >= 1).

        # From the derivation sqrt(Delta) >= 3k^2 + 6k (which holds if k_cubic_poly <= N
        # and delta is a perfect square), we have numerator = sqrt(Delta) - 3k^2 >= (3k^2 + 6k) - 3k^2 = 6k.
        # Since k >= 1, 6k > 0. Thus, numerator >= 6k > 0 is guaranteed under these conditions.
        # This means y = numerator / denominator >= 6k / 6k = 1.
        # So y will be a positive integer (y >= 1) if the numerator is divisible by the denominator.

        # We only need to check if the numerator is divisible by the denominator.
        # The condition numerator >= 0 is implicitly true if numerator % denominator == 0 and denominator > 0,
        # because the result y must be >= 1. However, explicitly checking it doesn't hurt.
        # The check y > 0 is also implicitly true if numerator % denominator == 0 and denominator > 0,
        # because numerator >= 6k > 0.

        if numerator >= 0 and numerator % denominator == 0:
             y = numerator // denominator # Calculate the integer value of y
             
             # At this point, y is guaranteed to be an integer >= 1.
             # We have found a positive integer solution y for the current k.
             # Calculate the corresponding x value using x = y + k.
             x = y + k

             # We have found a pair of positive integers (x, y) such that x^3 - y^3 = N.
             # Print x and y separated by a space and terminate the program.
             print(x, y)
             return # Exit the function and the program

        # If y is not an integer for this k, continue to the next value of k.
        k += 1

    # If the loop finishes without finding any solution for any k,
    # it means no pair of positive integers (x,y) exists such that x^3 - y^3 = N.
    # Print -1.
    print(-1)

# Execute the solve function to run the program.
solve()