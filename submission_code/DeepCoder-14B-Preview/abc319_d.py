def compute_lines(w, words):
    if not words:
        return 0
    current = words[0]
    lines = 1
    for word in words[1:]:
        if current + 1 + word <= w:
            current += 1 + word
        else:
            lines += 1
            current = word
    return lines

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    m = int(input[1])
    words = list(map(int, input[2:2+n]))
    
    if n == 0:
        print(0)
        return
    
    max_word = max(words)
    total_sum = sum(words) + (n - 1)
    
    low = max_word
    high = total_sum
    answer = high
    
    while low <= high:
        mid = (low + high) // 2
        lines = compute_lines(mid, words)
        if lines <= m:
            answer = mid
            high = mid - 1
        else:
            low = mid + 1
    print(answer)

if __name__ == "__main__":
    main()