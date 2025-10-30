def solve():
    n, m = map(int, input().split())
    l = list(map(int, input().split()))
    
    def count_lines(width, word_widths):
        if not word_widths:
            return 0
        lines_count = 1
        current_line_width = 0
        for word_width in word_widths:
            if current_line_width == 0:
                current_line_width = word_width
            else:
                if current_line_width + 1 + word_width <= width:
                    current_line_width += 1 + word_width
                else:
                    lines_count += 1
                    current_line_width = word_width
        return lines_count
        
    min_possible_width = 0
    low = max(l)
    high = sum(l) + n - 1 if n > 0 else 0
    result_width = -1
    
    while low <= high:
        mid_width = (low + high) // 2
        lines_needed = count_lines(mid_width, l)
        if lines_needed <= m:
            result_width = mid_width
            high = mid_width - 1
        else:
            low = mid_width + 1
            
    print(result_width)

if __name__ == '__main__':
    solve()