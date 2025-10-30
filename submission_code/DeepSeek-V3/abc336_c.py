# YOUR CODE HERE
def get_nth_good_integer(N):
    # The good integers can be mapped to numbers in base 5, where each digit represents an even digit (0,2,4,6,8)
    # For example, 0 -> 0, 1 -> 2, 2 -> 4, 3 -> 6, 4 -> 8
    # So, we can treat N as a base 5 number and map each digit to the corresponding even digit
    if N == 0:
        return 0
    result = []
    while N > 0:
        remainder = (N - 1) % 5
        result.append(str(2 * remainder))
        N = (N - 1) // 5
    result.reverse()
    return ''.join(result)

# Read input
N = int(input())
# Get the N-th good integer
good_int = get_nth_good_integer(N)
# Print the result
print(good_int)