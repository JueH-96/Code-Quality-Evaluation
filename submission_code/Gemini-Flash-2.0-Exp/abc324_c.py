def solve():
    n, t_prime = input().split()
    n = int(n)
    s_list = []
    for _ in range(n):
        s_list.append(input())
    
    possible_indices = []
    for i, s in enumerate(s_list):
        if is_possible(s, t_prime):
            possible_indices.append(i + 1)
    
    print(len(possible_indices))
    print(*possible_indices)

def is_possible(s, t_prime):
    # Case 1: s == t_prime
    if s == t_prime:
        return True
    
    # Case 2: t_prime is obtained by inserting one char in s
    if len(t_prime) == len(s) + 1:
        for i in range(len(s) + 1):
            temp = s[:i] + t_prime[i] + s[i:]
            if temp == t_prime:
                return True
    
    # Case 3: t_prime is obtained by deleting one char from s
    if len(t_prime) == len(s) - 1:
        for i in range(len(s)):
            temp = s[:i] + s[i+1:]
            if temp == t_prime:
                return True
    
    # Case 4: t_prime is obtained by changing one char in s
    if len(t_prime) == len(s):
        diff_count = 0
        for i in range(len(s)):
            if s[i] != t_prime[i]:
                diff_count += 1
        if diff_count == 1:
            return True
    
    return False

solve()