def solve():
    n = int(input())
    s = input()
    
    distinct_substrings = set()
    
    for i in range(n):
        for j in range(i, n):
            substring = s[i:j+1]
            if len(substring) > 0 and all(c == substring[0] for c in substring):
                distinct_substrings.add(substring)
                
    print(len(distinct_substrings))

solve()