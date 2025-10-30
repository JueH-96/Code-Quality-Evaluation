def solve():
    s = input()
    n = len(s)
    
    if n % 2 != 0:
        print("No")
        return
    
    for i in range(0, n, 2):
        if s[i] != s[i+1]:
            print("No")
            return
    
    counts = {}
    for char in s:
        counts[char] = counts.get(char, 0) + 1
    
    for count in counts.values():
        if count != 2:
            print("No")
            return
    
    print("Yes")

solve()