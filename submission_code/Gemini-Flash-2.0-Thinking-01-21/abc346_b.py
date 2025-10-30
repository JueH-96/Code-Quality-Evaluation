def solve():
    w, b = map(int, input().split())
    pattern = "wbwbwwbwbwbw"
    pattern_len = len(pattern)
    target_w_count = w
    target_b_count = b
    substring_length = target_w_count + target_b_count
    
    if substring_length == 0:
        if target_w_count == 0 and target_b_count == 0:
            print("Yes")
        else:
            print("No")
        return
        
    found_substring = False
    for start_index_1based in range(1, pattern_len + 1):
        current_w_count = 0
        current_b_count = 0
        substring = []
        for i in range(substring_length):
            current_index_0based = (start_index_1based - 1 + i) % pattern_len
            char = pattern[current_index_0based]
            substring.append(char)
            if char == 'w':
                current_w_count += 1
            elif char == 'b':
                current_b_count += 1
                
        if current_w_count == target_w_count and current_b_count == target_b_count:
            found_substring = True
            break
            
    if found_substring:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    solve()