def solve():
    k = int(input())
    s = input()
    t = input()
    
    if s == t:
        print("Yes")
        return
        
    possible = False
    
    # Insertion
    if len(t) == len(s) + 1:
        for i in range(len(s) + 1):
            for char_code in range(ord('a'), ord('z') + 1):
                char_to_insert = chr(char_code)
                s_prime = s[:i] + char_to_insert + s[i:]
                if s_prime == t:
                    possible = True
                    break
            if possible:
                break
                
    if possible:
        print("Yes")
        return
        
    # Deletion
    if len(t) == len(s) - 1:
        for i in range(len(s)):
            s_prime = s[:i] + s[i+1:]
            if s_prime == t:
                possible = True
                break
                
    if possible:
        print("Yes")
        return
        
    # Replacement
    if len(t) == len(s):
        for i in range(len(s)):
            for char_code in range(ord('a'), ord('z') + 1):
                char_to_replace = chr(char_code)
                s_prime = s[:i] + char_to_replace + s[i+1:]
                if s_prime == t:
                    possible = True
                    break
            if possible:
                break
                
    if possible:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    solve()