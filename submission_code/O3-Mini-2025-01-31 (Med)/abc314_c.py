def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    s = list(next(it))  # Convert string S to a list of characters for in-place modifications.
    colors = [int(next(it)) for _ in range(n)]
    
    # Group indices of S by their colors. Using 1-indexed colors, so we create a list of size m+1.
    indices_by_color = [[] for _ in range(m+1)]
    for i, color in enumerate(colors):
        indices_by_color[color].append(i)
    
    # For each color 1 through m (in order), perform a right circular shift on the characters at the specified indices.
    for color in range(1, m+1):
        pos_list = indices_by_color[color]
        if len(pos_list) <= 1:
            continue  # A single or no element means no effective shift.
        last_char = s[pos_list[-1]]
        # Shift characters to the right circularly.
        for j in range(len(pos_list)-1, 0, -1):
            s[pos_list[j]] = s[pos_list[j-1]]
        s[pos_list[0]] = last_char
    
    sys.stdout.write("".join(s))

if __name__ == '__main__':
    main()