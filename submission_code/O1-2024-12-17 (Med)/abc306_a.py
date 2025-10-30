def main():
    import sys
    # Read the inputs
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1]
    
    # Construct the doubled string
    answer = "".join(char * 2 for char in S)
    
    # Print the result
    print(answer)


# Call main function
main()