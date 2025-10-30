def main():
    import sys
    # Read input from standard input, splitting into parts
    input_data = sys.stdin.read().strip().split()
    # The first element is the surname
    surname = input_data[0]
    # Concatenate surname with " san" and print
    print(surname + " san")

# Call the main function
if __name__ == '__main__':
    main()