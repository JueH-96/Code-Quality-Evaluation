def can_be_original(s, t_prime):
    len_s = len(s)
    len_t = len(t_prime)
    
    # Case 1: No change
    if s == t_prime:
        return True
    
    # Case 2: One character inserted (t_prime is longer by 1)
    if len_t == len_s + 1:
        # Try removing each character from t_prime and see if it matches s
        for i in range(len_t):
            if t_prime[:i] + t_prime[i+1:] == s:
                return True
    
    # Case 3: One character deleted (t_prime is shorter by 1)
    if len_t == len_s - 1:
        # Try removing each character from s and see if it matches t_prime
        for i in range(len_s):
            if s[:i] + s[i+1:] == t_prime:
                return True
    
    # Case 4: One character changed (same length)
    if len_t == len_s:
        diff_count = 0
        for i in range(len_s):
            if s[i] != t_prime[i]:
                diff_count += 1
                if diff_count > 1:
                    break
        if diff_count == 1:
            return True
    
    return False

# Read input
line = input().split()
n = int(line[0])
t_prime = line[1]

strings = []
for _ in range(n):
    strings.append(input().strip())

# Find all strings that could be the original
valid_indices = []
for i in range(n):
    if can_be_original(strings[i], t_prime):
        valid_indices.append(i + 1)  # 1-indexed

# Output
print(len(valid_indices))
if valid_indices:
    print(*valid_indices)