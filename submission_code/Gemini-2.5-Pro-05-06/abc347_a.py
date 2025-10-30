# YOUR CODE HERE
def solve():
    # Read N and K from the first line of input.
    # N is the number of elements in sequence A, K is the divisor.
    # We use _N to indicate that N is not explicitly used in this Pythonic solution,
    # as the length of A can be implicitly determined.
    _N, K = map(int, input().split())
    
    # Read the sequence A from the second line of input.
    A = list(map(int, input().split()))
    
    # Create a list to store the quotients.
    quotients = []
    
    # Iterate through each element in the sequence A.
    for x_val in A:
        # Check if the element is a multiple of K.
        if x_val % K == 0:
            # If it is, divide it by K and add the quotient to our list.
            quotients.append(x_val // K)
            
    # The problem states that the input sequence A is sorted in strictly ascending order.
    # Since K is a positive integer, the quotients derived from A will also be
    # in strictly ascending order. Thus, the 'quotients' list is already sorted.
    
    # Print the elements of the 'quotients' list, separated by spaces.
    # The * operator unpacks the list, and print() separates arguments by space by default.
    print(*quotients)

# Call the solve function to execute the logic.
if __name__ == '__main__':
    solve()