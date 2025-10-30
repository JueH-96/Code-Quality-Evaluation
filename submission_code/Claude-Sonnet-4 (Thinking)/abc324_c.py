def can_be_transformed(T_prime, S):
    # Case 1: T' == S (no change)
    if T_prime == S:
        return True
    
    # Case 2: T' is obtained by inserting one character into S
    # This means len(T') == len(S) + 1
    if len(T_prime) == len(S) + 1:
        # Try removing each character from T' and see if result equals S
        for i in range(len(T_prime)):
            if T_prime[:i] + T_prime[i+1:] == S:
                return True
    
    # Case 3: T' is obtained by deleting one character from S
    # This means len(T') == len(S) - 1
    if len(T_prime) == len(S) - 1:
        # Try removing each character from S and see if result equals T'
        for i in range(len(S)):
            if S[:i] + S[i+1:] == T_prime:
                return True
    
    # Case 4: T' is obtained by changing one character in S
    # This means len(T') == len(S)
    if len(T_prime) == len(S):
        # Count the number of differing characters
        diff_count = 0
        for i in range(len(T_prime)):
            if T_prime[i] != S[i]:
                diff_count += 1
                if diff_count > 1:
                    return False
        return diff_count == 1
    
    return False

# Read input
first_line = input().split()
N = int(first_line[0])
T_prime = first_line[1]

valid_indices = []
for i in range(N):
    S = input().strip()
    if can_be_transformed(T_prime, S):
        valid_indices.append(i + 1)  # 1-indexed

# Output
print(len(valid_indices))
if valid_indices:
    print(*valid_indices)