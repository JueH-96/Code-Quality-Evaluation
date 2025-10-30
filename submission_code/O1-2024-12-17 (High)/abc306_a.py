# YOUR CODE HERE
def main():
    # Read integer N
    N = int(input().strip())
    # Read string S
    S = input().strip()
    
    # Build the result by repeating each character twice
    result = []
    for char in S:
        result.append(char * 2)
    
    # Print the concatenated result
    print("".join(result))

# Do not forget to call main()!
main()