def is_valid(s1, s2):
    """Check if s1 can be transformed into s2 by one of the four operations."""
    if s1 == s2:
        return True
    if len(s1) == len(s2) - 1:
        for i in range(len(s2)):
            if s1 == s2[:i] + s2[i+1:]:
                return True
    if len(s1) == len(s2) + 1:
        for i in range(len(s1)):
            if s2 == s1[:i] + s1[i+1:]:
                return True
    if len(s1) == len(s2):
        diff = 0
        for i in range(len(s1)):
            if s1[i] != s2[i]:
                diff += 1
            if diff > 1:
                break
        if diff == 1:
            return True
    return False

def solve():
    """Read input and solve the problem."""
    n = int(input())
    t_prime = input()
    s = []
    for _ in range(n):
        s.append(input())
    
    valid_indices = []
    for i in range(n):
        if is_valid(s[i], t_prime):
            valid_indices.append(i + 1)
    
    print(len(valid_indices))
    print(*valid_indices)

if __name__ == "__main__":
    solve()