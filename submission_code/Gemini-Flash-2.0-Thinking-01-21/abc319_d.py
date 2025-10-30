def solve():
    n, m = map(int, input().split())
    l = list(map(int, input().split()))
    
    def get_lines_needed(width):
        lines_count = 1
        current_line_width = 0
        for i in range(n):
            word_width = l[i]
            if current_line_width == 0:
                current_line_width = word_width
            else:
                if current_line_width + 1 + word_width <= width:
                    current_line_width += 1 + word_width
                else:
                    lines_count += 1
                    current_line_width = word_width
        return lines_count
        
    low = max(l) if l else 0
    high = sum(l) + max(0, n - 1)
    if n == 0:
        print(0)
        return
        
    min_width = high
    
    while low <= high:
        mid_width = (low + high) // 2
        lines_needed = get_lines_needed(mid_width)
        if lines_needed <= m:
            min_width = mid_width
            high = mid_width - 1
        else:
            low = mid_width + 1
            
    print(min_width)

if __name__ == '__main__':
    solve()