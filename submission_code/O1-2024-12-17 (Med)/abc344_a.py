def main():
    S = input().strip()
    # Find the indices of the two '|' characters
    first_bar = S.find('|')
    second_bar = S.rfind('|')
    
    # Remove the characters between (and including) the two '|'s
    result = S[:first_bar] + S[second_bar+1:]
    
    # Print the result
    print(result)

# Call the main function
main()