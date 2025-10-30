def common_prefix_length(s1, s2):
    length = 0
    for i in range(min(len(s1), len(s2))):
        if s1[i] == s2[i]:
            length += 1
        else:
            break
    return length

n = int(input())
strings = []
for _ in range(n):
    strings.append(input())

for k in range(1, n + 1):
    T = strings[k - 1]  # S_k (0-indexed)
    
    # Cost to make T empty
    min_cost = len(T)
    
    # Cost to make T match each previous string
    for i in range(k - 1):
        S_i = strings[i]
        common_len = common_prefix_length(T, S_i)
        cost = len(T) + len(S_i) - 2 * common_len
        min_cost = min(min_cost, cost)
    
    print(min_cost)