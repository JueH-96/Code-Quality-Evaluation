def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    # First input is N, second input is S
    n = int(input_data[0])
    s = input_data[1]
    
    # Construct the answer by duplicating each character
    result = ''.join(char * 2 for char in s)
    
    # Print the result
    sys.stdout.write(result)

if __name__ == '__main__':
    main()