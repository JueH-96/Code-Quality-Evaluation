import sys # Import sys module, although input() is typically sufficient for small inputs

# Define the main function to encapsulate the program logic
def main():
    # Initialize a counter variable to store the number of strings
    # satisfying the condition.
    count = 0
    
    # Iterate 12 times, with the index `i` taking values from 1 to 12.
    # This corresponds to reading strings S_1 through S_12.
    for i in range(1, 13):
        # Read the i-th string from standard input.
        # The input() function reads a line from stdin and removes the trailing newline character.
        s_i = input()
        
        # Calculate the length of the string read.
        length_s_i = len(s_i)
        
        # Check if the length of the string S_i is equal to its index i.
        if length_s_i == i:
            # If the condition is true, increment the counter.
            count += 1
            
    # After the loop finishes (i.e., after processing all 12 strings),
    # print the final value of the counter to standard output.
    print(count)

# This block ensures that the main function is called only when the script
# is executed directly (not when imported as a module).
if __name__ == '__main__':
    main()