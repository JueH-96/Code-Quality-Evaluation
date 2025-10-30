import sys

def longest_common_prefix_length(s1, s2):
    i = 0
    while i < len(s1) and i < len(s2) and s1[i] == s2[i]:
        i += 1
    return i

def solve_problem():
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    strings = data[1:]
    
    prefix_counts = {}
    total_sum = 0
    
    for s in strings:
        prefix = ''
        for char in s:
            prefix += char
            if prefix in prefix_counts:
                total_sum += prefix_counts[prefix]
                prefix_counts[prefix] += 1
            else:
                prefix_counts[prefix] = 1
    
    print(total_sum)

solve_problem()