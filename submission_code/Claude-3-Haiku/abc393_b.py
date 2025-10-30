import sys

def count_abc_triples(s):
    count = 0
    for i in range(len(s)-2):
        if s[i] == 'A':
            for j in range(i+1, len(s)):
                if s[j] == 'B':
                    for k in range(j+1, len(s)):
                        if s[k] == 'C' and j - i == k - j:
                            count += 1
    return count

s = input()
print(count_abc_triples(s))