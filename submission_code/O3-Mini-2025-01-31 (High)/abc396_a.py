def main():
    import sys

    # Read the entire input, split by whitespace and convert them into a list of integers.
    data = sys.stdin.read().split()
    if not data:
        return
    
    # The first number is N, the length of the sequence.
    N = int(data[0])
    
    # The next N numbers form the sequence A.
    A = [int(x) for x in data[1:1+N]]
    
    # Check for any three consecutive equal elements.
    for i in range(N - 2):
        if A[i] == A[i+1] == A[i+2]:
            print("Yes")
            return
    
    print("No")

# Call the main function to execute the solution.
if __name__ == '__main__':
    main()