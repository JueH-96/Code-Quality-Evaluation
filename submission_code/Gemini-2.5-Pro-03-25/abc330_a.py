import sys

def solve():
    # Read the first line: N and L
    n, l = map(int, sys.stdin.readline().split())
    
    # Read the second line: scores A_1, A_2, ..., A_N
    a = list(map(int, sys.stdin.readline().split()))
    
    # Initialize a counter for passed students
    pass_count = 0
    
    # Iterate through each score in the list a
    for score in a:
        # Check if the score is greater than or equal to the passing threshold L
        if score >= l:
            # If it is, increment the pass counter
            pass_count += 1
            
    # Print the final count of passed students
    print(pass_count)

# Call the solve function to execute the logic
solve()