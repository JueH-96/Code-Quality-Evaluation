def solve():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    S = data[0]
    Q = int(data[1])
    K = list(map(int, data[2].split()))
    
    n = len(S)
    
    # The length of S after 10^100 operations
    # After 1 operation, length becomes 2 * n
    # After 2 operations, length becomes 2 * (2 * n) = 4 * n
    # After k operations, length becomes 2^k * n
    # After 10^100 operations, length becomes 2^(10^100) * n
    # This is an extremely large number, but we only need to find the K_i-th character.
    
    # We will use the properties of the string and the operations to find the K_i-th character.
    
    # Precompute the transformation of S to T
    T = ''.join(c.lower() if c.isupper() else c.upper() for c in S)
    
    # We will find the K_i-th character by understanding the structure of the final string.
    results = []
    
    for k in K:
        # We need to find the character at position k (1-indexed)
        # We will reduce k to find its position in the original S or T
        while k > n:
            # Determine the current length of the string after some operations
            # The length of the string after `ops` operations is 2^ops * n
            # We need to find which segment k falls into
            length = n
            
            # We will find the largest power of 2 that is less than or equal to k
            ops = 0
            while length < k:
                ops += 1
                length *= 2
            
            # Now we know that k is in the string formed after `ops` operations
            # The string is formed by S + T, which means:
            # - First half is S (1 to n)
            # - Second half is T (n+1 to 2n)
            if k <= length // 2:
                # k is in the first half, which corresponds to S
                # Reduce k to find the character in S
                pass  # k remains the same
            else:
                # k is in the second half, which corresponds to T
                # Reduce k to find the character in T
                k -= length // 2  # Move to the second half
                # Now we need to find the corresponding character in T
            
            # If we are in T, we need to find the corresponding character in S
            # T is the transformation of S
            # If k is in T, we can find the corresponding character in S
            if k <= n:
                # k is in T, find the corresponding character in S
                # T[j] corresponds to S[j] transformed
                # T[j] = S[j].lower() if S[j].isupper() else S[j].upper()
                # We need to find the original character in S
                k -= 1  # Convert to 0-indexed
                if S[k].isupper():
                    results.append(S[k].lower())
                else:
                    results.append(S[k].upper())
            else:
                # This case should not happen as we are reducing k correctly
                raise ValueError("Unexpected value of k")
        
        # If k is still within the original string length
        k -= 1  # Convert to 0-indexed
        results.append(S[k])
    
    print(' '.join(results))