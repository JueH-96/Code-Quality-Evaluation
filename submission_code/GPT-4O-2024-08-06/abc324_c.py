def can_be_original(T_prime, S):
    len_T_prime = len(T_prime)
    len_S = len(S)
    
    if len_S == len_T_prime:
        # Check for equality or one character change
        diff_count = sum(1 for a, b in zip(T_prime, S) if a != b)
        return diff_count == 0 or diff_count == 1
    
    elif len_S + 1 == len_T_prime:
        # Check if T' can be obtained by inserting one character into S
        for i in range(len_T_prime):
            if T_prime[:i] + T_prime[i+1:] == S:
                return True
        return False
    
    elif len_S == len_T_prime + 1:
        # Check if T' can be obtained by deleting one character from S
        for i in range(len_S):
            if S[:i] + S[i+1:] == T_prime:
                return True
        return False
    
    return False

def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N, T_prime = data[0].split()
    N = int(N)
    
    results = []
    
    for i in range(1, N + 1):
        S = data[i]
        if can_be_original(T_prime, S):
            results.append(i)
    
    print(len(results))
    if results:
        print(' '.join(map(str, results)))