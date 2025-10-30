def solve():
    n = int(input())
    s = input()
    if n == 0:
        print(0)
        return
    
    max_run_lengths = {}
    i = 0
    while i < n:
        current_char = s[i]
        j = i
        while j < n and s[j] == current_char:
            j += 1
        run_length = j - i
        if current_char not in max_run_lengths:
            max_run_lengths[current_char] = 0
        max_run_lengths[current_char] = max(max_run_lengths[current_char], run_length)
        i = j
        
    count = 0
    for char in max_run_lengths:
        count += max_run_lengths[char]
        
    print(count)

if __name__ == '__main__':
    solve()