# YOUR CODE HERE
import sys

def solve():
    """
    Reads an integer N from standard input, constructs the Dragon String of level N,
    and prints it to standard output.
    """
    try:
        # Read the integer N from standard input
        N_str = sys.stdin.readline()
        # Handle potential empty input or errors during conversion
        if not N_str:
            # Or raise an error, depending on desired behavior for empty input
            return 
        N = int(N_str.strip()) 

        # Validate the input based on constraints (optional but good practice)
        if not (1 <= N <= 2024):
            # Handle invalid input range if necessary, though constraints usually guarantee valid input
            # For this problem, we can assume valid input based on the prompt.
            pass 

        # Construct the Dragon String:
        # Starts with 'L'
        # Followed by N occurrences of 'o'
        # Followed by 'n'
        # Ends with 'g'
        
        # Using string multiplication for the 'o' part and concatenation
        # dragon_string = 'L' + ('o' * N) + 'n' + 'g'
        
        # Alternatively, using an f-string (often more readable)
        dragon_string = f"L{'o' * N}ng"

        # Print the result to standard output
        print(dragon_string)

    except ValueError:
        # Handle the case where the input is not a valid integer
        # In a competitive programming context, inputs usually match the format,
        # so extensive error handling might not be needed unless specified.
        # print("Invalid input: N must be an integer.", file=sys.stderr)
        pass
    except Exception as e:
        # Catch any other unexpected errors
        # print(f"An error occurred: {e}", file=sys.stderr)
        pass


# Ensure the solve function runs when the script is executed
if __name__ == "__main__":
    solve()
# YOUR CODE HERE