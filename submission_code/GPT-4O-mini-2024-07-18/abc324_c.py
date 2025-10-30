def is_valid_insert(t_prime, s):
    # Check if t_prime can be obtained by inserting one character in s
    if len(t_prime) != len(s) + 1:
        return False
    for i in range(len(t_prime)):
        if t_prime[:i] + t_prime[i+1:] == s:
            return True
    return False

def is_valid_delete(t_prime, s):
    # Check if t_prime can be obtained by deleting one character from s
    if len(t_prime) != len(s) - 1:
        return False
    for i in range(len(s)):
        if s[:i] + s[i+1:] == t_prime:
            return True
    return False

def is_valid_change(t_prime, s):
    # Check if t_prime can be obtained by changing one character in s
    if len(t_prime) != len(s):
        return False
    diff_count = sum(1 for a, b in zip(t_prime, s) if a != b)
    return diff_count == 1

def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0].split()[0])
    t_prime = data[0].split()[1]
    S = data[1:N+1]
    
    valid_indices = []
    
    for i in range(N):
        s = S[i]
        if (s == t_prime or
            is_valid_insert(t_prime, s) or
            is_valid_delete(t_prime, s) or
            is_valid_change(t_prime, s)):
            valid_indices.append(i + 1)
    
    print(len(valid_indices))
    if valid_indices:
        print(' '.join(map(str, valid_indices)))

if __name__ == "__main__":
    main()