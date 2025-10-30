def solve():
    n, k = map(int, input().split())
    s = input()
    
    blocks = []
    i = 0
    while i < n:
        if s[i] == '1':
            start = i
            while i < n and s[i] == '1':
                i += 1
            blocks.append((start, i - 1))
        else:
            i += 1
            
    k -= 1
    
    if k >= len(blocks):
        print(s)
        return

    res = ""
    
    for i in range(blocks[k-1][1] + 1):
        res += s[i]
        
    for i in range(blocks[k][1] - blocks[k][0] + 1):
        res += '1'
        
    for i in range(blocks[k-1][1] + 1, blocks[k][0]):
        res += '0'

    for i in range(blocks[k][1] + 1, n):
        res += s[i]
        
    print(res)

solve()