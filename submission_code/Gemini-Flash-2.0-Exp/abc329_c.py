def solve():
    n = int(input())
    s = input()
    
    substrings = set()
    
    for i in range(n):
        for j in range(i, n):
            sub = s[i:j+1]
            if len(sub) > 0 and all(c == sub[0] for c in sub):
                substrings.add(sub)
    
    print(len(substrings))

solve()