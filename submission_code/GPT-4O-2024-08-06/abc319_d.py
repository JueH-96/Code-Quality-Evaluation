def can_fit_in_m_lines(words, max_width, m):
    current_width = 0
    lines_used = 1
    
    for word in words:
        if current_width + word <= max_width:
            current_width += word + 1  # Add word and a space
        else:
            lines_used += 1
            current_width = word + 1  # Start new line with this word
        
        if lines_used > m:
            return False
    
    return True

def find_minimum_width(n, m, words):
    low = max(words)
    high = sum(words) + (n - 1)
    
    while low < high:
        mid = (low + high) // 2
        if can_fit_in_m_lines(words, mid, m):
            high = mid
        else:
            low = mid + 1
    
    return low

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    m = int(data[1])
    words = list(map(int, data[2:]))
    
    result = find_minimum_width(n, m, words)
    print(result)