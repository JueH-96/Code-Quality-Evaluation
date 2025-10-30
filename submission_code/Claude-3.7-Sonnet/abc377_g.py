def min_cost(target, other_strings):
    # Cost to make the target empty
    min_cost = len(target)
    
    for s in other_strings:
        # Find the length of the common prefix
        common_prefix_len = 0
        for a, b in zip(target, s):
            if a == b:
                common_prefix_len += 1
            else:
                break
        
        # Calculate the cost to transform target into s
        # Cost = (characters to delete from target) + (characters to add from s)
        cost = (len(target) - common_prefix_len) + (len(s) - common_prefix_len)
        
        min_cost = min(min_cost, cost)
    
    return min_cost

N = int(input())
strings = []
for _ in range(N):
    strings.append(input())

for k in range(1, N+1):
    target = strings[k-1]
    other_strings = strings[:k-1]
    print(min_cost(target, other_strings))