# YOUR CODE HERE
import sys

def solve():
    """
    Reads an integer N and prints the N-th smallest good integer.
    A good integer consists only of even digits {0, 2, 4, 6, 8}.
    The solution maps this problem to a base-5 conversion.
    """
    try:
        # Read the integer N from standard input.
        n = int(sys.stdin.readline())
    except (ValueError, IndexError):
        # This case is not expected based on problem constraints.
        return

    # The N-th smallest good integer (1-indexed) corresponds to the
    # number N-1 (0-indexed). We convert N-1 to a base-5 representation
    # and then map each base-5 digit 'd' to '2*d'.
    k = n - 1

    # An empty list to store the digits of the resulting good integer.
    # The digits will be added in reverse order (least significant first).
    result_digits = []

    # This loop implements a "do-while" pattern, ensuring at least one
    # iteration. This correctly handles the edge case of N=1 (k=0),
    # where the result should be "0".
    while True:
        # Get the least significant digit of k in base 5.
        base5_digit = k % 5
        
        # Map the base-5 digit to its corresponding even digit.
        even_digit = base5_digit * 2
        
        # Add the resulting digit as a string to our list.
        result_digits.append(str(even_digit))
        
        # Perform integer division to process the next base-5 digit.
        k //= 5
        
        # If k becomes 0, we have processed all digits and can exit the loop.
        if k == 0:
            break

    # The digits were collected from least significant to most significant.
    # Reverse the list and join the characters to form the final number string.
    final_answer = "".join(reversed(result_digits))
    
    print(final_answer)

solve()