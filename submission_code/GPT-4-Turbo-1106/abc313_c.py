def min_operations_to_even_out(A):
    total_sum = sum(A)
    N = len(A)
    # Calculate the average, rounding down
    avg = total_sum // N
    # Calculate the number of elements that need to be increased to the average
    num_increase = sum(1 for x in A if x < avg)
    # Calculate the number of elements that need to be decreased to the average
    num_decrease = sum(1 for x in A if x > avg)
    # Calculate the remainder when dividing the total sum by N
    remainder = total_sum % N
    # The minimum number of operations is the maximum of the two numbers
    # plus the remainder, which represents the number of elements that
    # need to be adjusted by one more unit.
    return max(num_increase, num_decrease) + remainder

# Read input from stdin
N = int(input().strip())
A = list(map(int, input().strip().split()))

# Calculate and print the answer
print(min_operations_to_even_out(A))