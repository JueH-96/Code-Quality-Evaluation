K = int(input())
S = input().strip()
T = input().strip()

m, n = len(S), len(T)

# If length difference exceeds K, transformation is impossible
if abs(m - n) > K:
    print("No")
    exit()

# Banded edit distance computation
# Only compute cells where |i - j| <= K
prev = {}
for j in range(min(n + 1, K + 1)):
    prev[j] = j

for i in range(1, m + 1):
    curr = {}
    
    # Determine column range for this row
    j_min = max(0, i - K)
    j_max = min(n, i + K)
    
    for j in range(j_min, j_max + 1):
        if j == 0:
            if i <= K:
                curr[0] = i
        else:
            val = float('inf')
            
            # From diagonal (match/replace)
            if j - 1 in prev:
                cost = 0 if S[i-1] == T[j-1] else 1
                val = min(val, prev[j-1] + cost)
            
            # From above (delete from S)
            if j in prev:
                val = min(val, prev[j] + 1)
            
            # From left (insert into S)
            if j - 1 in curr:
                val = min(val, curr[j-1] + 1)
            
            curr[j] = val
    
    prev = curr

# Check if edit distance is at most K
if n in prev and prev[n] <= K:
    print("Yes")
else:
    print("No")