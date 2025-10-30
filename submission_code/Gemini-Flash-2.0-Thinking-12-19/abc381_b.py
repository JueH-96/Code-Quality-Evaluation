def solve():
    s = input()
    n = len(s)
    if n % 2 != 0:
        print("No")
        return
    
    for i in range(n // 2):
        if s[2*i] != s[2*i + 1]:
            print("No")
            return
            
    char_counts = {}
    for char in s:
        char_counts[char] = char_counts.get(char, 0) + 1
        
    for char in char_counts:
        if char_counts[char] != 2:
            print("No")
            return
            
    print("Yes")

if __name__ == '__main__':
    solve()