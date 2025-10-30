# YOUR CODE HERE
def longest_common_prefix_length(s1, s2):
    length = 0
    min_len = min(len(s1), len(s2))
    for i in range(min_len):
        if s1[i] == s2[i]:
            length += 1
        else:
            break
    return length

n = int(input())
strings = input().split()

total = 0
for i in range(n):
    for j in range(i + 1, n):
        total += longest_common_prefix_length(strings[i], strings[j])

print(total)