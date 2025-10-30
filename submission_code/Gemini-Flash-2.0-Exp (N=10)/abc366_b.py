def solve():
    n = int(input())
    s = [input() for _ in range(n)]
    
    max_len = 0
    for string in s:
        max_len = max(max_len, len(string))
    
    result = [""] * max_len
    
    for i in range(n):
        for j in range(len(s[i])):
            while len(result[j]) < n:
                result[j] += "*"
            result[j] = result[j][:n - i - 1] + s[i][j] + result[j][n - i:]
    
    for i in range(max_len):
        while result[i].endswith("*") and len(result[i]) > 0:
            result[i] = result[i][:-1]
        print(result[i])
        
solve()