import sys

def solve():
    # Read N
    N = int(sys.stdin.readline())
    
    # Read array A
    A = list(map(int, sys.stdin.readline().split()))

    # Calculate the total sum of elements
    total_sum = sum(A)

    # Determine the two possible target values (X and X+1)
    # The lower target value is S // N
    target_low_value = total_sum // N
    # The higher target value is (S // N) + 1
    target_high_value = target_low_value + 1

    # Determine how many elements should be the higher value
    # This is S % N
    num_high_targets = total_sum % N
    
    # The remaining elements should be the lower value
    num_low_targets = N - num_high_targets

    # To minimize operations, we should assign the smallest initial values
    # to the lower target value and the largest initial values to the higher target value.
    # This is achieved by sorting the array.
    A.sort()

    # Calculate the sum of absolute differences between current values and their assigned target values.
    # This sum, when divided by 2, gives the minimum number of operations.
    total_abs_diff = 0

    # The first 'num_low_targets' elements in the sorted array should become 'target_low_value'
    for i in range(num_low_targets):
        total_abs_diff += abs(A[i] - target_low_value)

    # The remaining 'num_high_targets' elements should become 'target_high_value'
    for i in range(num_low_targets, N):
        total_abs_diff += abs(A[i] - target_high_value)
    
    # Each unit moved from a higher value to a lower value counts as one operation.
    # The `total_abs_diff` calculates the sum of all excesses and all deficits.
    # Since total excess must equal total deficit, `total_abs_diff` is twice the actual operations.
    # So, we divide by 2.
    print(total_abs_diff // 2)

# Call the solve function to execute the program
solve()