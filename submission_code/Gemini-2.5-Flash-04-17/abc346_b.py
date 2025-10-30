# YOUR CODE HERE
import sys

# Read input W and B
W, B = map(int, sys.stdin.readline().split())

# The base pattern of the infinite keyboard
T = "wbwbwwbwbwbw"
T_len = 12

# Precompute prefix counts of 'w' and 'b' for the pattern T
# w_prefix_T[k] = number of 'w's in T[:k] (string slice from index 0 up to k-1)
# b_prefix_T[k] = number of 'b's in T[:k]
# We need counts for k from 0 to 12.
w_prefix_T = [0]
b_prefix_T = [0]
w_count_t = 0
b_count_t = 0
for char in T:
    if char == 'w':
        w_count_t += 1
    else:
        b_count_t += 1
    w_prefix_T.append(w_count_t)
    b_prefix_T.append(b_count_t)
# w_prefix_T and b_prefix_T now have length T_len + 1 = 13.
# w_prefix_T[k] stores the count in T[:k]. Index k ranges from 0 to 12.
# w_prefix_T[12] is the total count of 'w' in T (which is 7).
# b_prefix_T[12] is the total count of 'b' in T (which is 5).

# Length of the target substring
L = W + B

# Function to calculate count of 'w' in the first m characters of the infinite string S
# S[:m] consists of floor(m/T_len) full T patterns and T[:m%T_len]
# m is 0-indexed total length from the start of S.
def get_w_count_until(m):
    if m < 0:
        # This case should ideally not be reached for valid substring boundaries
        # but handle defensively. Count is 0 for negative length prefixes.
        return 0
    # Number of full repetitions of T within the first m characters
    num_full_t = m // T_len
    # Length of the remaining part (a prefix of T)
    remainder_len = m % T_len
    # Total count = count in full T patterns + count in the remaining prefix of T
    return num_full_t * w_prefix_T[T_len] + w_prefix_T[remainder_len]

# Function to calculate count of 'b' in the first m characters of the infinite string S
def get_b_count_until(m):
    if m < 0:
        # Handle defensively for negative length prefixes.
        return 0
    # Number of full repetitions of T within the first m characters
    num_full_t = m // T_len
    # Length of the remaining part (a prefix of T)
    remainder_len = m % T_len
    # Total count = count in full T patterns + count in the remaining prefix of T
    return num_full_t * b_prefix_T[T_len] + b_prefix_T[remainder_len]


# We need to check if there is a substring S[i : i+L] for *any* i >= 0.
# A substring S[i : i+L] consists of characters S[i], S[i+1], ..., S[i+L-1].
# Due to the periodicity S[k] = S[k+12], the sequence S[i : i+L] has the same
# character composition as S[i%12 : i%12+L].
# So, we only need to check substrings starting at indices i = 0, 1, ..., 11
# in the infinite string S.
# The count of 'w' in S[i : i+L] is the count of 'w' in the first (i+L) chars
# minus the count of 'w' in the first i chars.
# This is CountW(S[:i+L]) - CountW(S[:i]).

found = False
# Iterate through the 12 possible starting phases (indices 0 to 11) within the base pattern
for i in range(T_len):
    # Calculate count of 'w' and 'b' in the substring S[i : i+L]
    # The end index of the substring slice is i + L (exclusive).
    # The count in S[:i+L] gives counts up to index i+L-1.
    # The count in S[:i] gives counts up to index i-1.
    # The difference gives counts in S[i:i+L].
    current_w_count = get_w_count_until(i + L) - get_w_count_until(i)
    current_b_count = get_b_count_until(i + L) - get_b_count_until(i)

    # Check if the calculated counts match the required W and B
    if current_w_count == W and current_b_count == B:
        found = True
        break # Found a matching substring, no need to check further

# Output the result
if found:
    print("Yes")
else:
    print("No")