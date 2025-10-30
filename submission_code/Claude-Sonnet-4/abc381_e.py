def solve():
    N, Q = map(int, input().split())
    S = input().strip()
    
    for _ in range(Q):
        L, R = map(int, input().split())
        # Convert to 0-based indexing
        L -= 1
        R -= 1
        
        substring = S[L:R+1]
        
        # Find maximum length 11/22 string subsequence
        max_length = 0
        
        # Try different values of k (number of 1s and 2s each)
        # Maximum possible k is limited by the length of substring
        max_k = len(substring) // 2
        
        for k in range(max_k + 1):
            # Try to find k ones, 1 slash, k twos in order
            if can_form_11_22_string(substring, k):
                max_length = max(max_length, 2 * k + 1)
        
        print(max_length)

def can_form_11_22_string(s, k):
    # Need to find k ones, 1 slash, k twos in that order as subsequence
    n = len(s)
    
    # Find k ones
    ones_found = 0
    i = 0
    while i < n and ones_found < k:
        if s[i] == '1':
            ones_found += 1
        i += 1
    
    if ones_found < k:
        return False
    
    # Find 1 slash
    slash_found = False
    while i < n and not slash_found:
        if s[i] == '/':
            slash_found = True
        i += 1
    
    if not slash_found:
        return False
    
    # Find k twos
    twos_found = 0
    while i < n and twos_found < k:
        if s[i] == '2':
            twos_found += 1
        i += 1
    
    return twos_found == k

solve()