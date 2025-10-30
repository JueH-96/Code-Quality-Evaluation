def longest_common_prefix(s1, s2):
    lcp_length = 0
    for c1, c2 in zip(s1, s2):
        if c1 == c2:
            lcp_length += 1
        else:
            break
    return lcp_length

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
strings = data[1:]

total_lcp_sum = 0

for i in range(N - 1):
    for j in range(i + 1, N):
        total_lcp_sum += longest_common_prefix(strings[i], strings[j])

print(total_lcp_sum)