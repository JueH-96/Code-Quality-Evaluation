def main():
    import sys
    # Read the input string from standard input
    S = sys.stdin.readline().strip()
    
    # Find the index of the first '|' character
    first_index = S.find('|')
    # Find the index of the second '|' character, starting search after the first one
    second_index = S.find('|', first_index + 1)
    
    # Remove everything between the two '|' characters including the '|'s
    # This is the concatenation of the part before the first '|' and the part after the second '|'
    result = S[:first_index] + S[second_index + 1:]
    
    # Print the resulting string
    sys.stdout.write(result)

# Call the main function
if __name__ == '__main__':
    main()