def longest_common_prefix(s1, s2):
    count = 0
    min_len = min(len(s1), len(s2))
    for k in range(min_len):
        if s1[k] == s2[k]:
            count += 1
        else:
            break
    return count

n = int(input())
s = [input() for _ in range(n)]

total_lcp = 0
for i in range(n):
    for j in range(i + 1, n):
        total_lcp += longest_common_prefix(s[i], s[j])

print(total_lcp)