def solve():
    s = input()
    q = int(input())
    queries = list(map(int, input().split()))
    
    ans = []
    for k in queries:
        original_len = len(s)
        ops = 0
        temp_k = k
        while temp_k > original_len:
            temp_k = (temp_k + 1) // 2
            ops += 1
        
        idx = temp_k - 1
        char = s[idx]
        
        flip_count = 0
        
        while ops > 0:
            if k > original_len * (2**ops):
                
                if 'a' <= char <= 'z':
                    char = char.upper()
                else:
                    char = char.lower()
                
                flip_count +=1
            
            k = (k + 1) // 2
            ops -= 1
        
        
        if flip_count % 2 == 1:
            if 'a' <= char <= 'z':
                char = char.upper()
            else:
                char = char.lower()
        
        ans.append(char)
    
    print(*ans)

solve()