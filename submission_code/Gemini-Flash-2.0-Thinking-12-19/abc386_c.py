def solve():
    k = int(input())
    s = input()
    t = input()
    
    if s == t:
        print("Yes")
        return
        
    # Check for insertion
    if len(t) == len(s) + 1:
        for i in range(len(t)):
            s_prime = t[:i] + t[i+1:]
            if s_prime == s:
                print("Yes")
                return
                
    # Check for deletion
    if len(t) == len(s) - 1:
        for i in range(len(s)):
            t_prime = s[:i] + s[i+1:]
            if t_prime == t:
                print("Yes")
                return
                
    # Check for replacement
    if len(t) == len(s):
        diff_count = 0
        for i in range(len(s)):
            if s[i] != t[i]:
                diff_count += 1
        if diff_count == 1:
            print("Yes")
            return
            
    print("No")

if __name__ == '__main__':
    solve()