import math

# YOUR CODE HERE
R = int(input())

count = 0

# Calculate R^2 once to avoid repeated computation
R_sq = R * R

# Determine the maximum absolute value for i.
# A square centered at (i, j) is contained if (abs(i) + 0.5)^2 + (abs(j) + 0.5)^2 <= R^2.
# This implies (abs(i) + 0.5)^2 <= R^2, so abs(i) + 0.5 <= R.
# Thus, abs(i) <= R - 0.5.
# Since R is an integer, R - 0.5 is always of the form X.5 (e.g., 1.5, 3.5).
# floor(X.5) is X. So, floor(R - 0.5) is R - 1.
# The maximum possible integer value for abs(i) is R - 1.
max_abs_i = R - 1

# Iterate through possible integer values of i, from -(R-1) to (R-1) inclusive.
for i in range(-max_abs_i, max_abs_i + 1):
    i_abs = abs(i)
    
    # Calculate (abs(i) + 0.5)^2 in an integer-friendly way by scaling by 4.
    # (k + 0.5)^2 = ((2k+1)/2)^2 = (2k+1)^2 / 4
    # So, (abs(i) + 0.5)^2 corresponds to (2 * i_abs + 1)^2 / 4
    x_term_scaled_numerator = (2 * i_abs + 1)
    x_term_sq_scaled = x_term_scaled_numerator * x_term_scaled_numerator

    # Calculate the maximum allowed value for (abs(j) + 0.5)^2, also scaled by 4.
    # The core condition is (abs(i) + 0.5)^2 + (abs(j) + 0.5)^2 <= R^2.
    # Scaling by 4: x_term_sq_scaled + (2 * abs(j) + 1)^2 <= 4 * R^2
    # So, (2 * abs(j) + 1)^2 <= 4 * R^2 - x_term_sq_scaled
    max_y_term_sq_scaled = 4 * R_sq - x_term_sq_scaled

    # If this value is negative, it means no possible j can satisfy the condition for the current i.
    # For example, if (abs(i) + 0.5)^2 is already greater than R^2.
    if max_y_term_sq_scaled < 0:
        continue

    # Find the largest integer 's' such that s*s <= max_y_term_sq_scaled.
    # math.isqrt(N) computes floor(sqrt(N)) efficiently and accurately for integers.
    s = math.isqrt(max_y_term_sq_scaled)

    # We need (2 * abs(j) + 1) <= s (since (2*abs(j)+1)^2 <= s^2).
    # This implies 2 * abs(j) <= s - 1.
    # So, abs(j) <= (s - 1) / 2.
    # The maximum integer value for abs(j) is floor((s - 1) / 2).
    # Integer division // naturally handles the floor operation for positive results.
    max_abs_j = (s - 1) // 2

    # Count the number of j values for the current i.
    # If max_abs_j is negative, it means even j=0 (where abs(j)=0) does not satisfy the condition.
    # In this case, there are 0 valid j values.
    # Otherwise, j can be 0, +/-1, ..., +/-max_abs_j.
    # This gives 1 (for j=0) + 2 * max_abs_j (for positive and negative values).
    if max_abs_j < 0:
        num_j_values = 0
    else:
        num_j_values = 1 + 2 * max_abs_j
    
    count += num_j_values

print(count)