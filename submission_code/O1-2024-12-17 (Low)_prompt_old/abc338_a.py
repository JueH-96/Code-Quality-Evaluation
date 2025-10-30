def solve():
    S = input().strip()
    
    # Check if the first character is uppercase
    if not S[0].isupper():
        print("No")
        return
    
    # If the string has only one character, it's already uppercase, so "Yes"
    if len(S) == 1:
        print("Yes")
        return
    
    # Check if all remaining characters are lowercase
    if S[1:].islower():
        print("Yes")
    else:
        print("No")

# Call the solve function
solve()