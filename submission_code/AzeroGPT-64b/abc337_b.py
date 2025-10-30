def main():
    # Read a single line of input and store it in S
    S = input().strip()
    # Initialize variables to keep track of the current character
    last_char = None
    # Iterate over each character in the string
    for char in S:
        # If the current character is less than the last one (A before B, B before C)
        if last_char is not None and ord(char) < ord(last_char):
            # Print 'No' and return from the function, as the string is not an Extended ABC string
            print('No')
            return
        # Update the last character
        last_char = char
    # If the loop completes without returning, the string is an Extended ABC string
    print('Yes')

# Call the main function if the script is executed directly
if __name__ == '__main__':
    main()