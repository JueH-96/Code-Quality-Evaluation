def main():
    S = input().strip()
    # Find positions of the two '|'
    first_bar = S.find('|')
    second_bar = S.find('|', first_bar + 1)
    
    # Remove the characters between the two '|' (inclusive)
    result = S[:first_bar] + S[second_bar+1:]
    
    print(result)

# Do not remove the below function call
main()