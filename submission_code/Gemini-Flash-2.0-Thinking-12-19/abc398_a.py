# Read input N
N = int(input())

# Construct the string based on N's parity
if N % 2 != 0:
    # N is odd, exactly one '=' at the center
    # The center index is N // 2. There are N // 2 dashes before and after it.
    num_dashes = N // 2
    result_string = "-" * num_dashes + "=" + "-" * num_dashes
else:
    # N is even, exactly two adjacent '='s at the center
    # The two center indices are N // 2 - 1 and N // 2.
    # There are (N // 2 - 1) dashes before the first '=' and (N // 2 - 1) dashes after the second '='.
    # This handles N=2 correctly, where num_dashes = 2//2 - 1 = 0.
    num_dashes = N // 2 - 1
    result_string = "-" * num_dashes + "==" + "-" * num_dashes

# Print the result string
print(result_string)