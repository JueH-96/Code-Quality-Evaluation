def are_similar_chars(x, y):
    if x == y:
        return True
    if (x == '1' and y == 'l') or (x == 'l' and y == '1'):
        return True
    if (x == '0' and y == 'o') or (x == 'o' and y == '0'):
        return True
    return False

def solve():
    n = int(input())
    s = input()
    t = input()
    
    if len(s) != n or len(t) != n:
        raise ValueError("Input strings length mismatch with N")
        
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