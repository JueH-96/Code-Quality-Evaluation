def solve():
    n = int(input())
    s = input()
    distinct_substrings = set()
    for char_code in range(ord('a'), ord('z') + 1):
        char_to_check = chr(char_code)
        i = 0
        while i < n:
            if s[i] == char_to_check:
                j = i
                while j + 1 < n and s[j+1] == char_to_check:
                    j += 1
                for length in range(1, j - i + 2):
                    substring = s[i:i+length]
                    distinct_substrings.add(substring)
                i = j + 1
            else:
                i += 1
    print(len(distinct_substrings))

if __name__ == '__main__':
    solve()