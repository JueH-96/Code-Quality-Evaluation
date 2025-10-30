def solve():
    line = input().split()
    n = int(line[0])
    t = line[1]
    
    strings = []
    for _ in range(n):
        strings.append(input().strip())
    
    m = len(t)
    
    # For each string, precompute what prefixes of t can be matched
    # and what needs to be matched after each position
    prefix_match = []  # prefix_match[i][j] = can we match first j chars of t using strings[i]?
    
    for s in strings:
        can_match = [False] * (m + 1)
        can_match[0] = True
        
        j = 0
        for c in s:
            if j < m and c == t[j]:
                j += 1
                can_match[j] = True
        
        prefix_match.append(can_match)
    
    # Count valid pairs
    count = 0
    for i in range(n):
        for j in range(n):
            # Check if concatenation contains t as subsequence
            s = strings[i] + strings[j]
            k = 0
            for c in s:
                if k < m and c == t[k]:
                    k += 1
            if k == m:
                count += 1
    
    print(count)

solve()