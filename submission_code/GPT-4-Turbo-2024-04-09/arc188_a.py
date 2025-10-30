MOD = 998244353

def count_good_substrings(s):
    n = len(s)
    count = 0
    # Check for each possible substring
    for i in range(n):
        for j in range(i + 1, n + 1):
            substring = s[i:j]
            if is_good_string(substring):
                count += 1
    return count

def is_good_string(t):
    # Convert string to list for manipulation
    t = list(t)
    while t:
        changed = False
        # First try to apply Operation 1
        for char in 'ABC':
            if t.count(char) >= 2:
                # Remove two occurrences of char
                first = t.index(char)
                t.pop(first)
                second = t.index(char)
                t.pop(second)
                changed = True
                break
        if not changed:
            # Try to apply Operation 2
            if all(x in t for x in 'ABC'):
                t.remove('A')
                t.remove('B')
                t.remove('C')
                changed = True
        if not changed:
            return False
    return True

def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    K = int(data[1])
    S = data[2]
    
    # Generate all possible replacements for '?'
    from itertools import product
    
    replacements = []
    for i in range(N):
        if S[i] == '?':
            replacements.append(i)
    
    count = 0
    for comb in product('ABC', repeat=len(replacements)):
        # Create a new string with the current combination
        new_s = list(S)
        for idx, char in zip(replacements, comb):
            new_s[idx] = char
        new_s = ''.join(new_s)
        
        # Count good substrings in this configuration
        if count_good_substrings(new_s) >= K:
            count += 1
    
    print(count % MOD)