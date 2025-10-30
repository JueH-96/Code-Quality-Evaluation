def solve():
    s, t = input().split()
    n = len(s)
    target_t = t
    
    for w in range(1, n):
        for c in range(1, w + 1):
            substrings = []
            start_index = 0
            while start_index < n:
                end_index = min(start_index + w, n)
                substrings.append(s[start_index:end_index])
                start_index += w
            
            generated_t = ""
            for sub in substrings:
                if len(sub) >= c:
                    generated_t += sub[c-1]
                    
            if generated_t == target_t:
                print("Yes")
                return
                
    print("No")

if __name__ == '__main__':
    solve()