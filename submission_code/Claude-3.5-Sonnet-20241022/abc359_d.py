def is_palindrome(s):
    return s == s[::-1]

def check_valid(s, k):
    for i in range(len(s) - k + 1):
        if is_palindrome(s[i:i+k]):
            return False
    return True

def solve(n, k, s):
    MOD = 998244353
    
    # Get positions of question marks
    q_pos = []
    for i in range(n):
        if s[i] == '?':
            q_pos.append(i)
    
    # If no question marks, just check if original string is good
    if not q_pos:
        return 1 if check_valid(s, k) else 0
        
    # Try all possible combinations
    ans = 0
    q = len(q_pos)
    for mask in range(1 << q):
        # Create current string by replacing ? with A/B
        curr = list(s)
        for i in range(q):
            if mask & (1 << i):
                curr[q_pos[i]] = 'A'
            else:
                curr[q_pos[i]] = 'B'
        curr = ''.join(curr)
        
        # Check if current string is good
        if check_valid(curr, k):
            ans = (ans + 1) % MOD
            
    return ans

# Read input
n, k = map(int, input().split())
s = input().strip()

# Print answer
print(solve(n, k, s))