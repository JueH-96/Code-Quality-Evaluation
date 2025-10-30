def main():
    # Read the input and convert it to an integer.
    n = int(input().strip())
    
    # Concatenate n copies of the digit n and print the resulting string.
    result = str(n) * n
    print(result)

# Call the main function to execute the solution.
main()