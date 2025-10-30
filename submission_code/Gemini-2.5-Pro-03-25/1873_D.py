# YOUR CODE HERE
import sys

# Function to solve a single test case
def solve():
    # Read n and k from the first line of the test case.
    # Use sys.stdin.readline for potentially faster input reading compared to input().
    line1 = sys.stdin.readline().split()
    n = int(line1[0]) # Length of the paper strip
    k = int(line1[1]) # Length of the segment affected by one operation
    
    # Read the string s representing the paper strip from the second line.
    s = sys.stdin.readline().strip()
    
    # Initialize the count of operations needed.
    count = 0
    # Initialize the current index pointer 'i' to the beginning of the strip.
    i = 0
    
    # Iterate through the string using a while loop. The loop continues as long as
    # the index 'i' is within the bounds of the string (0 to n-1).
    while i < n:
        # Check if the cell at the current index 'i' is black ('B').
        if s[i] == 'B':
            # If the cell is black, we must perform an operation to make it white.
            # The problem asks for the minimum number of operations.
            # A greedy strategy works here: when we encounter the leftmost uncovered black cell,
            # we perform an operation that covers this cell and extends k positions (from i to i+k-1).
            # This operation makes the current black cell white, along with potentially other cells
            # up to index i+k-1.
            
            # Increment the operation count because we are performing one operation.
            count += 1
            
            # Since the operation covers cells from index i up to i+k-1, all these cells
            # become white. We don't need to worry about any black cells within this range
            # anymore with respect to requiring a *new* operation starting before i+k.
            # Therefore, we can safely jump our pointer 'i' forward by k positions.
            # The next potential black cell that might require a new operation must be at index i+k or later.
            i += k 
        else:
            # If the current cell s[i] is white ('W'), it doesn't need an operation.
            # We simply move to check the next cell in the strip.
            i += 1
    
    # After the loop finishes (when i >= n), 'count' holds the minimum number of operations required
    # to make all black cells white. Print the result for this test case.
    print(count)

# Read the number of test cases 't' from the first line of input.
t = int(sys.stdin.readline())
# Process each test case by calling the solve function 't' times.
for _ in range(t):
    solve()

# END OF YOUR CODE HERE