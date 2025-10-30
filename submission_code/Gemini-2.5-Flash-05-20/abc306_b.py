# YOUR CODE HERE
def solve():
    # Read the input line and split it into a list of strings
    a_str_list = input().split()

    # Convert the list of string digits to a list of integers
    # The map(int, ...) function applies int() to each element of the iterable
    a_int_list = list(map(int, a_str_list))

    # Initialize the total sum
    total_sum = 0

    # Iterate through the list of digits with their corresponding indices
    # A_i corresponds to a_int_list[i], and its power is 2^i
    for i in range(len(a_int_list)):
        # If the digit A_i is 1, add 2^i to the total sum
        # Using (1 << i) is an efficient way to calculate 2^i
        if a_int_list[i] == 1:
            total_sum += (1 << i)
            
    # Print the final calculated sum
    print(total_sum)

# Call the solve function to execute the logic
if __name__ == '__main__':
    solve()