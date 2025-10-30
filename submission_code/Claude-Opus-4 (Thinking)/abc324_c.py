# YOUR CODE HERE
def can_delete_one_to_match(longer, shorter):
    # Check if we can delete exactly one character from 'longer' to get 'shorter'
    if len(longer) != len(shorter) + 1:
        return False
    
    i = 0  # pointer for shorter
    skipped = False
    
    for j in range(len(longer)):
        if i < len(shorter) and longer[j] == shorter[i]:
            i += 1
        elif not skipped:
            skipped = True
        else:
            return False
    
    return i == len(shorter)

def could_be_T(s, t_prime):
    # Check no change
    if s == t_prime:
        return True
    
    # Check insertion (T' = T with one char inserted)
    if len(t_prime) == len(s) + 1 and can_delete_one_to_match(t_prime, s):
        return True
    
    # Check deletion (T' = T with one char deleted)
    if len(s) == len(t_prime) + 1 and can_delete_one_to_match(s, t_prime):
        return True
    
    # Check replacement
    if len(s) == len(t_prime):
        diff_count = 0
        for i in range(len(s)):
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

result = []
for i in range(1, n + 1):
    s = input().strip()
    if could_be_T(s, t_prime):
        result.append(i)

# Output
print(len(result))
if result:
    print(' '.join(map(str, result)))