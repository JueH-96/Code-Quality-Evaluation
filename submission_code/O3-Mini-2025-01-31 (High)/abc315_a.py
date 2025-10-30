def main():
    # Read the input string and remove trailing whitespace
    s = input().strip()
    
    # Define the set of vowels
    vowels = set("aeiou")
    
    # Build the result string by filtering out vowels
    result = "".join(char for char in s if char not in vowels)
    
    # Print the result
    print(result)

# Call main to execute the solution
if __name__ == "__main__":
    main()