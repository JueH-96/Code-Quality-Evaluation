# Read the two integers N and K from the first line of standard input.
# The values are not strictly needed for the logic if we use list comprehensions,
# but we read them to conform to the input format.
N, K = map(int, input().split())

# Read the sequence of N integers from the second line of standard input.
A = list(map(int, input().split()))

# Use a list comprehension to create a new list containing the required quotients.
# 1. `for x in A`: This iterates through each element `x` in the list `A`.
# 2. `if x % K == 0`: This is a filter. It checks if the element `x` is a multiple of `K`.
# 3. `x // K`: If the element is a multiple, it is divided by `K` using integer division.
# The resulting quotients are collected into a new list.
# Since the input list A is sorted, the list of quotients will also be sorted.
quotients = [x // K for x in A if x % K == 0]

# Print the elements of the `quotients` list to standard output.
# The `*` operator unpacks the list, passing its elements as separate arguments
# to the print function. By default, `print` separates arguments with a space.
print(*quotients)