def calculate_result():
    # Read input from stdin
    sequence = input().split()

    # Initialize result variable
    result = 0

    # Iterate over the sequence and calculate the result
    for i, bit in enumerate(sequence):
        if bit == '1':
            result += 2 ** i

    # Print the result
    print(result)

# Call the function
calculate_result()