def count_single_char_substrings(s):
    unique_substrings = set()
    i = 0
    while i < len(s):
        char = s[i]
        j = i
        while j < len(s) and s[j] == char:
            j += 1
        run_length = j - i
        for length in range(1, run_length + 1):
            unique_substrings.add((char, length))
        i = j
    return len(unique_substrings)

n = int(input())
s = input().strip()
print(count_single_char_substrings(s))