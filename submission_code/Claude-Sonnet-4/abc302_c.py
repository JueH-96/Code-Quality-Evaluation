from itertools import permutations

def differs_by_one(s1, s2):
    """Check if two strings differ by exactly one character"""
    if len(s1) != len(s2):
        return False
    
    diff_count = 0
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            diff_count += 1
            if diff_count > 1:
                return False
    
    return diff_count == 1

def can_form_sequence(strings):
    """Check if strings can be rearranged to form a valid sequence"""
    n = len(strings)
    
    # Try all possible permutations
    for perm in permutations(strings):
        valid = True
        # Check if this permutation satisfies the condition
        for i in range(n - 1):
            if not differs_by_one(perm[i], perm[i + 1]):
                valid = False
                break
        
        if valid:
            return True
    
    return False

# Read input
n, m = map(int, input().split())
strings = []
for _ in range(n):
    strings.append(input().strip())

# Solve and output
if can_form_sequence(strings):
    print("Yes")
else:
    print("No")