def main():
    S = input().strip()
    # Find the positions of the two '|' characters
    first = S.find('|')
    second = S.find('|', first + 1)
    # Concatenate the substring before the first '|' and after the second '|'
    result = S[:first] + S[second + 1:]
    print(result)

# Call the main function
main()