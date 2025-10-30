def solve():
    n = int(input())
    s = input()
    valid_substrings = set()
    i = 0
    while i < n:
        char = s[i]
        j = i
        while j < n and s[j] == char:
            substring = s[i : j + 1]
            valid_substrings.add(substring)
            j += 1
        i = j
    print(len(valid_substrings))

solve()