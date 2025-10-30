def main():
    import sys
    # Read input, which is provided on standard input.
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    A = int(input_data[0])
    
    # Check if 400 people can be arranged in A rows evenly.
    if 400 % A == 0:
        B = 400 // A
        print(B)
    else:
        print(-1)

# Calling the main function.
if __name__ == '__main__':
    main()