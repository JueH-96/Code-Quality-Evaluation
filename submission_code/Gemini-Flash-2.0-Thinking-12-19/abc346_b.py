def solve():
    w, b = map(int, input().split())
    pattern = "wbwbwwbwbwbw"
    pattern_len = len(pattern)
    prefix_len = 24 + w + b
    s_prefix = ""
    for i in range(prefix_len):
        s_prefix += pattern[i % pattern_len]
    
    found = False
    for i in range(len(s_prefix)):
        for j in range(i, len(s_prefix)):
            substring = s_prefix[i:j+1]
            w_count = 0
            b_count = 0
            for char in substring:
                if char == 'w':
                    w_count += 1
                elif char == 'b':
                    b_count += 1
            if w_count == w and b_count == b:
                found = True
                break
        if found:
            break
            
    if found:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    solve()