def main():
    # Read the input string S
    S = input().strip()
    
    # Change the last character to '4'
    result = S[:-1] + '4'
    
    # Print the modified string
    print(result)

# Call main to execute the solution
main()