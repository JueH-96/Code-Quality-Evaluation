import sys

# Read N and S from standard input.
N = int(sys.stdin.readline())
S = sys.stdin.readline().strip()

# The problem asks for the sum of values of all substrings of S.
# Let S = s_0 s_1 ... s_{N-1} where s_k is the digit at index k (0-indexed).
# A substring S[i..j] (0-indexed) corresponds to indices i through j.
# The value of this substring is f(i+1, j+1) = sum_{k=i}^j int(s_k) * 10^(j-k).

# We want to compute the total sum: Sum = sum_{i=0}^{N-1} sum_{j=i}^{N-1} f(i+1, j+1).

# Let's analyze the contribution of each digit s_k at index k (0-indexed) to the total sum.
# The digit s_k appears in any substring S[i..j] where 0 <= i <= k <= j < N.
# In such a substring, the value of s_k contributes int(s_k) * 10^(j-k) to f(i+1, j+1).
# The exponent (j-k) is the position of s_k from the right end of the substring (0-indexed).

# Let's fix the digit s_k at index k (0 <= k < N).
# We sum its contribution int(s_k) * 10^(j-k) over all substrings S[i..j] that contain s_k.
# The valid ranges for i and j are 0 <= i <= k and k <= j < N.
# Total contribution of s_k = sum_{i=0}^k sum_{j=k}^{N-1} int(s_k) * 10^(j-k).

# We can factor out int(s_k):
# Total contribution of s_k = int(s_k) * sum_{i=0}^k sum_{j=k}^{N-1} 10^(j-k).
# The inner sums are independent.
# sum_{i=0}^k 1 = k+1 (this represents the number of possible starting indices i for a fixed k and j).
# sum_{j=k}^{N-1} 10^(j-k). Let p = j-k. When j=k, p=0. When j=N-1, p=N-1-k.
# sum_{j=k}^{N-1} 10^(j-k) = sum_{p=0}^{N-1-k} 10^p = 1 + 10 + ... + 10^(N-1-k).
# Let's define T'_m = sum_{p=0}^m 10^p. So the sum is T'_{N-1-k}.

# The total contribution of digit s_k is int(S[k]) * (k+1) * T'_{N-1-k}.
# The total sum of all substrings is the sum of contributions of all digits from k=0 to N-1:
# Total Sum = sum_{k=0}^{N-1} int(S[k]) * (k+1) * T'_{N-1-k}.

# We need the values T'_{N-1}, T'_{N-2}, ..., T'_0 for k=0, 1, ..., N-1 respectively.
# We can compute T'_m using the recurrence: T'_0 = 1, T'_m = T'_{m-1} * 10 + 1 for m >= 1.
# We can calculate T'_{N-1} first using the forward recurrence starting from T'_0.
# Then, we can find the preceding values T'_{m-1} from T'_m using the reverse relation: T'_{m-1} = (T'_m - 1) // 10.
# This allows us to compute the sum in O(N) time and O(1) additional space.

# Calculate T'_{N-1}.
# Initialize T'_0 = 1.
T_N_minus_1 = 1
# If N > 1, compute T'_1, T'_2, ..., T'_{N-1}.
# The loop iterates N-1 times for N > 1, 0 times for N = 1.
for _ in range(N - 1):
    T_N_minus_1 = T_N_minus_1 * 10 + 1

# Initialize the variable that will hold T'_{N-1-k} for the current k.
# For the first iteration k=0, we need T'_{N-1}.
T_prime_needed = T_N_minus_1

total_sum = 0

# Iterate through the digits of S from left to right (0-indexed k from 0 to N-1).
# We apply the formula: Total Sum = sum_{k=0}^{N-1} int(S[k]) * (k+1) * T'_{N-1-k}.
for k in range(N):
    digit = int(S[k])

    # (k+1) is the coefficient representing the number of times int(S[k]) * 10^p
    # contributes to the sum for a fixed power p = j-k (or fixed end index j).
    # This comes from summing over the possible starting indices i (0 to k).
    multiplier_based_on_start_pos = k + 1

    # T_prime_needed currently holds the value T'_{N-1-k}. This is the sum of powers of 10
    # corresponding to all possible positions p from the right (0 to N-1-k).
    # The term to add for digit S[k] is int(S[k]) * (k+1) * T'_{N-1-k}.
    term = digit * multiplier_based_on_start_pos * T_prime_needed
    total_sum += term

    # Update T_prime_needed for the next iteration k+1.
    # The index for T' needed for k+1 is N-1-(k+1) = N-2-k.
    # If the current index for T' is m = N-1-k, the next index needed is m-1.
    # We have T'_m (in T_prime_needed) and need T'_{m-1} = (T'_m - 1) // 10.
    # This update is only performed if there is a next iteration (i.e., k < N-1).
    if k < N - 1:
        T_prime_needed = (T_prime_needed - 1) // 10

# Print the final calculated sum. Python's arbitrary precision integers handle large values.
print(total_sum)