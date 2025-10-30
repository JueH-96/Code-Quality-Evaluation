import sys

def count_substrings(s):
    count = 0
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and s[j] == s[i]:
            j += 1
        count += (j - i) * (j - i + 1) // 2
        i = j
    return count

N = int(sys.stdin.readline().strip())
S = sys.stdin.readline().strip()

print(count_substrings(S))