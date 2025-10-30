def solve():
    n, c1, c2 = input().split()
    n = int(n)
    s = input()
    
    result = ""
    for char in s:
        if char == c1:
            result += char
        else:
            result += c2
    print(result)

solve()