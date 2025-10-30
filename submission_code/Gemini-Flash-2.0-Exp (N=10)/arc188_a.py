def is_good(s):
    while True:
        changed = False
        counts = {}
        for char in s:
            counts[char] = counts.get(char, 0) + 1
        
        for char in counts:
            if counts[char] >= 2:
                s = s.replace(char*2, '', 1)
                changed = True
                break
        if changed:
            continue
        
        if 'A' in counts and 'B' in counts and 'C' in counts:
            s = s.replace('A', '', 1)
            s = s.replace('B', '', 1)
            s = s.replace('C', '', 1)
            changed = True
        
        if not changed:
            break
    return len(s) == 0

def solve():
    n, k = map(int, input().split())
    s = input()
    
    q_count = s.count('?')
    ans = 0
    
    for i in range(3**q_count):
        temp = list(s)
        temp_i = i
        
        q_idx = 0
        for j in range(n):
            if temp[j] == '?':
                if temp_i % 3 == 0:
                    temp[j] = 'A'
                elif temp_i % 3 == 1:
                    temp[j] = 'B'
                else:
                    temp[j] = 'C'
                temp_i //= 3
        
        temp_s = "".join(temp)
        
        good_count = 0
        for start in range(n):
            for end in range(start, n):
                sub = temp_s[start:end+1]
                if is_good(sub):
                    good_count += 1
        
        if good_count >= k:
            ans += 1
    
    print(ans % 998244353)

solve()