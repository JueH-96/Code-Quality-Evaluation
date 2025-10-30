def solve():
    n = int(input())
    s = input()

    for i in range(n):
        prefix = s[:i+1]
        chars = set()
        for char in prefix:
            chars.add(char)
        if len(chars) == 3:
            print(i + 1)
            return