def is_good_string(s):
    # Count frequencies of each character
    freq = {'A': 0, 'B': 0, 'C': 0}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    
    while True:
        # Try Operation 2 first (ABC)
        if freq['A'] >= 1 and freq['B'] >= 1 and freq['C'] >= 1:
            freq['A'] -= 1
            freq['B'] -= 1
            freq['C'] -= 1
            continue
        
        # Try Operation 1 (pairs)
        found_pair = False
        for c in ['A', 'B', 'C']:
            if freq[c] >= 2:
                freq[c] -= 2
                found_pair = True
                break
        
        if not found_pair:
            break
    
    # If all characters are used up, it's a good string
    return sum(freq.values()) == 0

def count_good_substrings(s):
    n = len(s)
    count = 0
    for i in range(n):
        for j in range(i+1, n+1):
            if is_good_string(s[i:j]):
                count += 1
    return count

def solve(N, K, S):
    MOD = 998244353
    
    # Count question marks and their positions
    q_positions = []
    for i in range(N):
        if S[i] == '?':
            q_positions.append(i)
    
    # If no question marks, just check if the string meets the requirement
    if not q_positions:
        count = count_good_substrings(S)
        return 1 if count >= K else 0
    
    # Try all possible combinations
    result = 0
    choices = ['A', 'B', 'C']
    q_count = len(q_positions)
    
    for i in range(3 ** q_count):
        # Generate one possible combination
        current = list(S)
        temp = i
        for pos in q_positions:
            current[pos] = choices[temp % 3]
            temp //= 3
        
        # Count good substrings in this combination
        if count_good_substrings(''.join(current)) >= K:
            result = (result + 1) % MOD
    
    return result

# Read input
N, K = map(int, input().split())
S = input().strip()

# Print output
print(solve(N, K, S))