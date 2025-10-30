def find_nth_good_integer(N):
    result = []
    while N > 0:
        remainder = (N - 1) % 5
        result.append(str(2 * remainder))
        N = (N - 1) // 5
    return ''.join(reversed(result)) if result else '0'

# Read input
N = int(input())
# Compute and print the result
print(find_nth_good_integer(N))