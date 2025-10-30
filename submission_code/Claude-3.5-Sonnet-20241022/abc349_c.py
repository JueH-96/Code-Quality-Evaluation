s = input().strip()
t = input().strip()

def is_airport_code(s, t):
    # Case 1: Three letter subsequence
    n = len(s)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if t == (s[i] + s[j] + s[k]).upper():
                    return True
                    
    # Case 2: Two letter subsequence + X
    if t[2] == 'X':
        for i in range(n):
            for j in range(i+1, n):
                if t[:2] == (s[i] + s[j]).upper():
                    return True
    
    return False

print("Yes" if is_airport_code(s, t) else "No")