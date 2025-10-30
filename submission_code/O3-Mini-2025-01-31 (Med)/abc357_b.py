def main():
    # Read input string S from standard input
    S = input().strip()
    
    # Count the number of uppercase and lowercase letters in the string
    count_upper = sum(1 for ch in S if ch.isupper())
    count_lower = len(S) - count_upper  # since S only contains letters
    
    # If the number of uppercase letters is greater than the number of lowercase letters,
    # convert all lowercase letters in S to uppercase.
    # Otherwise, convert all uppercase letters in S to lowercase.
    if count_upper > count_lower:
        result = S.upper()
    else:
        result = S.lower()
    
    # Print the resulting string
    print(result)

# Call the main function
if __name__ == '__main__':
    main()