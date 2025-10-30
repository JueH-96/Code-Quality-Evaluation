def are_similar_chars(char1, char2):
    if char1 == char2:
        return True
    if (char1 == '1' and char2 == 'l') or (char1 == 'l' and char2 == '1'):
        return True
    if (char1 == '0' and char2 == 'o') or (char1 == 'o' and char2 == '0'):
        return True
    return False

def solve():
    n = int(input())
    s = input()
    t = input()
    
    similar_strings = True
    for i in range(n):
        if not are_similar_chars(s[i], t[i]):
            similar_strings = False
            break
            
    if similar_strings:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    solve()