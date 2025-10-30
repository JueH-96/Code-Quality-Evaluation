def are_characters_similar(char1, char2):
    # Characters are similar if they are the same
    if char1 == char2:
        return True
    
    # Characters are similar if one is '1' and the other is 'l'
    if (char1 == '1' and char2 == 'l') or (char1 == 'l' and char2 == '1'):
        return True
    
    # Characters are similar if one is '0' and the other is 'o'
    if (char1 == '0' and char2 == 'o') or (char1 == 'o' and char2 == '0'):
        return True
    
    # If none of the above, characters are not similar
    return False

def are_strings_similar(s, t):
    for i in range(len(s)):
        if not are_characters_similar(s[i], t[i]):
            return False
    return True

# Read input
N = int(input())
S = input()
T = input()

# Check if strings are similar and output result
if are_strings_similar(S, T):
    print("Yes")
else:
    print("No")