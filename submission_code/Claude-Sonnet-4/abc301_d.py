S = input().strip()
N = int(input().strip())

def backtrack(pos, current_str):
    if pos == len(S):
        val = int(current_str, 2) if current_str else 0
        return val if val <= N else -1
    
    if S[pos] != '?':
        return backtrack(pos + 1, current_str + S[pos])
    
    # Try '1' first (greedy for maximum)
    result = backtrack(pos + 1, current_str + '1')
    if result != -1:
        return result
    
    # Try '0'
    return backtrack(pos + 1, current_str + '0')

result = backtrack(0, "")
print(result)