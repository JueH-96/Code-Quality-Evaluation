# Read the number of elements, N.
# While N is given, we can also infer the length from the list of numbers.
# However, reading it explicitly follows the problem's input format.
try:
    n = int(input())
except (ValueError, EOFError):
    # Handle cases with no input gracefully, although problem constraints guarantee valid input.
    n = 0

# Read the sequence of N space-separated integers into a list 'a'.
# The map() function applies int() to each item from input().split(),
# and list() collects the results into a list.
if n > 0:
    a = list(map(int, input().split()))
else:
    a = []

# The condition for a strictly increasing sequence is A_i < A_{i+1} for all valid i.
# We can check the opposite: if we find any i where A_i >= A_{i+1}, the sequence
# is not strictly increasing.

# We iterate from the first element up to the second-to-last element.
# The range(n - 1) generates indices from 0 to n-2.
for i in range(n - 1):
    # Check if the current element is greater than or equal to the next one.
    if a[i] >= a[i+1]:
        # If this condition is met, we have found a counter-example.
        # The sequence is not strictly increasing. Print "No" and terminate the script.
        print("No")
        exit()

# If the loop completes without finding any counter-examples (i.e., without exiting),
# it means that A_i < A_{i+1} was true for all pairs.
# Therefore, the sequence is strictly increasing.
print("Yes")