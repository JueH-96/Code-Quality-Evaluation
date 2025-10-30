import sys

# Read N
N = int(sys.stdin.readline())

# Read A as a list of integers
A = list(map(int, sys.stdin.readline().split()))

operations_count = 0

# Helper function to count the number of positive elements in the list
def count_positive_elements(arr):
    return sum(1 for x in arr if x > 0)

# The simulation continues until A contains one or fewer positive elements.
while True:
    # Check the termination condition
    # If there are 0 or 1 positive elements, we stop.
    if count_positive_elements(A) <= 1:
        break

    # Step 1: Sort A in descending order.
    # This brings the largest elements (which are guaranteed to be positive
    # if count_positive_elements(A) > 1) to the beginning of the list.
    A.sort(reverse=True)

    # Step 2: Decrease A_1 (A[0]) and A_2 (A[1]) by 1.
    # Since count_positive_elements(A) > 1, it means there are at least two positive numbers.
    # After sorting, these two positive numbers will be A[0] and A[1] (as A_i are non-negative after operations).
    A[0] -= 1
    A[1] -= 1

    # Increment the operation count
    operations_count += 1

# Print the total number of operations performed
print(operations_count)