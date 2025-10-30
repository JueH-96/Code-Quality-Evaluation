def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    # Group elements by their f-value
    special_group = 0  # for f-value 1/2 (elements with a=1 or a=2)
    other_groups = {}  # for other f-values (elements with a>=3)
    
    for val in a:
        if val == 1 or val == 2:
            special_group += 1
        else:
            other_groups[val] = other_groups.get(val, 0) + 1
    
    result = 0
    
    # Pairs from the special group
    if special_group >= 2:
        result += special_group * (special_group - 1) // 2
    
    # Pairs from other groups
    for cnt in other_groups.values():
        if cnt >= 2:
            result += cnt * (cnt - 1) // 2
    
    return result

t = int(input())
for _ in range(t):
    print(solve())