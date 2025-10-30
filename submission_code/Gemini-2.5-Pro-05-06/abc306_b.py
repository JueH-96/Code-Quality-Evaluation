def main():
    # Read the sequence A_0, A_1, ..., A_63 as a list of integers.
    # input().split() reads the line and splits it into a list of strings.
    # map(int, ...) converts each string in that list to an integer.
    # list(...) converts the map object into a list.
    A = list(map(int, input().split()))

    # Initialize the total sum.
    total_sum = 0

    # Initialize the current power of 2. This will represent 2^i.
    # It starts at 2^0 = 1 for A_0.
    current_power_of_2 = 1

    # Iterate through the elements of the list A.
    # val_A_i will be A_0 in the first iteration, A_1 in the second, and so on.
    for val_A_i in A:
        # Add the term (A_i * 2^i) to the total sum.
        # Since val_A_i is constrained to be 0 or 1, this effectively means
        # adding current_power_of_2 to total_sum if val_A_i is 1,
        # or adding 0 if val_A_i is 0.
        total_sum += val_A_i * current_power_of_2
        
        # Update current_power_of_2 for the next element A_{i+1}.
        # This changes it from 2^i to 2^{i+1}.
        current_power_of_2 *= 2
        
    # Print the final calculated sum as an integer.
    print(total_sum)

if __name__ == '__main__':
    main()