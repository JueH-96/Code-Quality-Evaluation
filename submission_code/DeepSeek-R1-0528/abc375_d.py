def main():
    S = input().strip()
    from collections import defaultdict
    char_positions = defaultdict(list)
    for index, char in enumerate(S):
        char_positions[char].append(index)
    
    ans = 0
    for char, positions in char_positions.items():
        m = len(positions)
        if m < 2:
            continue
        suffix = [0] * (m + 1)
        for i in range(m - 1, -1, -1):
            suffix[i] = suffix[i + 1] + positions[i]
        for i in range(m - 1):
            t = m - i - 1
            ans += suffix[i + 1] - positions[i] * t - t
    print(ans)

if __name__ == '__main__':
    main()