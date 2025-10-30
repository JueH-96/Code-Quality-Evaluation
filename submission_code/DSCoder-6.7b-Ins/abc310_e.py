def solve():
    N = int(input().strip())
    S = input().strip()

    # Convert the string to a list of integers
    A = [int(s) for s in S]

    # Initialize the sum to 0
    total_sum = 0

    # Initialize the current value to 0
    current_value = 0

    # Iterate over the list in reverse order
    for i in range(N-1, -1, -1):
        # Update the current value
        current_value = current_value ^ A[i]
        # Add the current value to the total sum
        total_sum += current_value
        # If the current value is 1, add N to the total sum
        if current_value == 1:
            total_sum += N

    # Print the total sum
    print(total_sum)

solve()