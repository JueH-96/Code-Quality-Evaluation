def minStartingIndex(s: str, p: str) -> int:
    m = len(p)
    n = len(s)
    if m == 0 or n < m:
        return -1
    
    # Calculate the initial sum of mismatches for the first window
    sum_prev = 0
    for j in range(m):
        if s[j] != p[j]:
            sum_prev += 1
    
    if sum_prev <= 1:
        return 0
    
    # Slide the window
    for i in range(1, n - m + 1):
        # Remove the contribution of the character leaving the window
        sum_prev -= (s[i-1] != p[0])
        # Add the contribution of the new character entering the window
        sum_prev += (s[i + m - 1] != p[m - 1])
        
        if sum_prev <= 1:
            return i
    
    return -1