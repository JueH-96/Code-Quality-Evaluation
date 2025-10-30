def main():
    # Read the input string
    S = input().strip()
    # Extract the first character and concatenate it with "UPC"
    result = S[0] + "UPC"
    # Print the result
    print(result)

# Call the main function to execute the program
if __name__ == "__main__":
    main()