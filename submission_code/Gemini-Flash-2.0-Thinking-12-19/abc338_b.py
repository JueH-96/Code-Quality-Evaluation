def solve():
    s = input()
    counts = {}
    for char in s:
        counts[char] = counts.get(char, 0) + 1

    max_freq = 0
    for char in counts:
        max_freq = max(max_freq, counts[char])

    max_freq_chars = []
    for char in counts:
        if counts[char] == max_freq:
            max_freq_chars.append(char)

    max_freq_chars.sort()
    print(max_freq_chars[0])

if __name__ == "__main__":
    solve()