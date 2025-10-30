def main():
    import sys
    input_data = sys.stdin.read().split()
    n = int(input_data[0])
    # Read lists A and B from the input_data
    A = list(map(int, input_data[1:1+n]))
    B = list(map(int, input_data[1+n:1+2*n]))
    
    # The maximum achievable sum is the sum of the maximum elements from A and B
    result = max(A) + max(B)
    
    # Print the result
    sys.stdout.write(str(result))
    
if __name__ == '__main__':
    main()