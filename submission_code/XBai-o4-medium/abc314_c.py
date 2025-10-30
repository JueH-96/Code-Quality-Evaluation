def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    n = int(input[idx])
    idx += 1
    m = int(input[idx])
    idx += 1
    s = list(input[idx])
    idx += 1
    c = list(map(int, input[idx:idx + n]))
    idx += n
    
    color_positions = [[] for _ in range(m + 1)]  # 1-based indexing for colors
    for pos in range(1, n + 1):
        color = c[pos - 1]
        color_positions[color].append(pos)
    
    for color in range(1, m + 1):
        pos_list = color_positions[color]
        k = len(pos_list)
        if k <= 1:
            continue
        # Collect old characters
        old_chars = [s[pos - 1] for pos in pos_list]
        new_chars = [old_chars[-1]] + old_chars[:-1]
        # Update the string
        for j in range(k):
            s[pos_list[j] - 1] = new_chars[j]
    
    print(''.join(s))

if __name__ == '__main__':
    main()