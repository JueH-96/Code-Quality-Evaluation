def solve():
    n = int(input())
    s = input()

    for i in range(1, n + 1):
        prefix = s[:i]
        seen_chars = set()
        for char in prefix:
            seen_chars.add(char)
        if 'A' in seen_chars and 'B' in seen_chars and 'C' in seen_chars:
            print(i)
            return

solve()