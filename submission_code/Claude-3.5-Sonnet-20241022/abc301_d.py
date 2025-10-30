S = input()
N = int(input())

def solve(s, n):
    # Count number of question marks
    q_count = s.count('?')
    if q_count == 0:
        val = int(s, 2)
        return val if val <= n else -1
        
    # Get all possible values by replacing ? with 0/1
    ans = -1
    for i in range(1 << q_count):
        curr = list(s)
        pos = 0
        for j in range(len(curr)):
            if curr[j] == '?':
                curr[j] = '1' if (i & (1 << pos)) else '0'
                pos += 1
        val = int(''.join(curr), 2)
        if val <= n:
            ans = max(ans, val)
    return ans

print(solve(S, N))