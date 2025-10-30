# YOUR CODE HERE
def longest_common_prefix(s1, s2):
    n = min(len(s1), len(s2))
    for i in range(n):
        if s1[i] != s2[i]:
            return i
    return n

n = int(input())
strings = []
for _ in range(n):
    strings.append(input())

for k in range(n):
    min_cost = len(strings[k])  # Cost to make empty
    
    for i in range(k):
        lcp = longest_common_prefix(strings[k], strings[i])
        cost = len(strings[k]) + len(strings[i]) - 2 * lcp
        min_cost = min(min_cost, cost)
    
    print(min_cost)