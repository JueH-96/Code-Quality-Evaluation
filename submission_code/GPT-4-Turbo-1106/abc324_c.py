def could_be_equal(t_prime, s):
    # Check if T' is equal to S
    if t_prime == s:
        return True
    # Check if T' is obtained by inserting one character in S
    if len(t_prime) - len(s) == 1:
        for i in range(len(t_prime)):
            if t_prime[:i] + t_prime[i+1:] == s:
                return True
    # Check if T' is obtained by deleting one character from S
    if len(s) - len(t_prime) == 1:
        for i in range(len(s)):
            if s[:i] + s[i+1:] == t_prime:
                return True
    # Check if T' is obtained by changing one character in S
    if len(t_prime) == len(s):
        diff_count = sum(1 for a, b in zip(t_prime, s) if a != b)
        if diff_count == 1:
            return True
    return False

# Read input
n = int(input().strip())
t_prime = input().strip()
strings = [input().strip() for _ in range(n)]

# Find all strings that could be equal to T
possible_indices = [i + 1 for i, s in enumerate(strings) if could_be_equal(t_prime, s)]

# Output the result
print(len(possible_indices))
if len(possible_indices) > 0:
    print(' '.join(map(str, possible_indices)))