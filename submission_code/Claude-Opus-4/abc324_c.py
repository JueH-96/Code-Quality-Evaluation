# YOUR CODE HERE
def can_be_original(s, t_prime):
    len_s = len(s)
    len_t = len(t_prime)
    
    # Case 1: T' equals T (s equals t_prime)
    if s == t_prime:
        return True
    
    # Case 2: T' is T with one character inserted (t_prime is s with one char inserted)
    if len_t == len_s + 1:
        # Try to match by skipping one character in t_prime
        i = 0
        skipped = False
        for j in range(len_t):
            if i < len_s and s[i] == t_prime[j]:
                i += 1
            elif not skipped:
                skipped = True
            else:
                return False
        return i == len_s
    
    # Case 3: T' is T with one character deleted (t_prime is s with one char deleted)
    if len_t == len_s - 1:
        # Try to match by skipping one character in s
        j = 0
        skipped = False
        for i in range(len_s):
            if j < len_t and s[i] == t_prime[j]:
                j += 1
            elif not skipped:
                skipped = True
            else:
                return False
        return j == len_t
    
    # Case 4: T' is T with one character changed (t_prime is s with one char changed)
    if len_t == len_s:
        diff_count = 0
        for i in range(len_s):
            if s[i] != t_prime[i]:
                diff_count += 1
                if diff_count > 1:
                    return False
        return diff_count == 1
    
    return False

# Read input
n, t_prime = input().split()
n = int(n)

# Check each string
valid_indices = []
for i in range(1, n + 1):
    s = input().strip()
    if can_be_original(s, t_prime):
        valid_indices.append(i)

# Output
print(len(valid_indices))
if valid_indices:
    print(' '.join(map(str, valid_indices)))