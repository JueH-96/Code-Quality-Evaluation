def longest_common_prefix_length(s1, s2):
    i = 0
    while i < len(s1) and i < len(s2) and s1[i] == s2[i]:
        i += 1
    return i

def transform_cost(source, target):
    if not target:  # target is empty string
        return len(source)
    
    common_len = longest_common_prefix_length(source, target)
    # Delete suffix of source + add suffix of target
    return len(source) - common_len + len(target) - common_len

n = int(input())
strings = []
for _ in range(n):
    strings.append(input().strip())

for k in range(1, n + 1):
    current = strings[k - 1]  # S_k (0-indexed)
    
    # Option 1: make T empty
    min_cost = len(current)
    
    # Option 2: make T match one of S_1, S_2, ..., S_{k-1}
    for i in range(k - 1):  # i from 0 to k-2 (S_1 to S_{k-1})
        target = strings[i]
        cost = transform_cost(current, target)
        min_cost = min(min_cost, cost)
    
    print(min_cost)