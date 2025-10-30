# YOUR CODE HERE
def rearrange_digits():
    import sys
    input = sys.stdin.read
    N = input().strip()
    
    # Extract digits
    a = N[0]
    b = N[1]
    c = N[2]
    
    # Form the two new numbers
    bca = b + c + a
    cab = c + a + b
    
    # Print the results
    print(bca, cab)

rearrange_digits()