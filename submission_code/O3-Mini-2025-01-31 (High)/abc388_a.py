def main():
    # Read the input string and strip any extra whitespace/newlines
    S = input().strip()
    # Concatenate the first character of S with "UPC"
    result = S[0] + "UPC"
    # Print the result
    print(result)

# Call the main function
if __name__ == '__main__':
    main()