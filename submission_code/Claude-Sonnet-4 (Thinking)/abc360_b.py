def solve():
    S, T = input().split()
    
    n = len(S)
    
    for w in range(1, n):
        for c in range(1, w + 1):
            result = ""
            for i in range(0, n, w):
                chunk = S[i:i+w]
                if len(chunk) >= c:
                    result += chunk[c-1]
            
            if result == T:
                return "Yes"
    
    return "No"

print(solve())