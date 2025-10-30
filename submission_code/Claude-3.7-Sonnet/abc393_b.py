def count_abc_triples(s):
    count = 0
    n = len(s)
    
    for i in range(n):
        for d in range(1, (n - i) // 2 + 1):
            j = i + d
            k = i + 2 * d
            
            if k < n and s[i] == 'A' and s[j] == 'B' and s[k] == 'C':
                count += 1
    
    return count

# Read input
S = input().strip()
print(count_abc_triples(S))