def could_be_original(s, t_prime):
    # Check if s could be T (the string Takahashi sent), given that Aoki received t_prime
    
    if s == t_prime:  # Condition 1: T' is equal to T
        return True
    
    if len(s) == len(t_prime) - 1:  # Condition 2: T' is T with one character inserted
        # Check if t_prime with one character removed is s
        for i in range(len(t_prime)):
            if t_prime[:i] + t_prime[i+1:] == s:
                return True
    
    if len(s) == len(t_prime) + 1:  # Condition 3: T' is T with one character deleted
        # Check if s with one character removed is t_prime
        for i in range(len(s)):
            if s[:i] + s[i+1:] == t_prime:
                return True
    
    if len(s) == len(t_prime):  # Condition 4: T' is T with one character changed
        # Check if s and t_prime differ in exactly one position
        diff_count = 0
        for i in range(len(s)):
            if s[i] != t_prime[i]:
                diff_count += 1
            if diff_count > 1:
                return False
        return diff_count == 1
    
    return False

# Read the input
n, t_prime = input().split()
n = int(n)
strings = []
for _ in range(n):
    strings.append(input())

# Find the original strings
originals = []
for i, s in enumerate(strings):
    if could_be_original(s, t_prime):
        originals.append(i + 1)  # +1 because the indices are 1-based

# Print the result
print(len(originals))
if originals:
    print(' '.join(map(str, originals)))