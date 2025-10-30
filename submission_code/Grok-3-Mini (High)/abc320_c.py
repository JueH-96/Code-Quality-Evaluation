import sys
import collections

# Read input
data = sys.stdin.read().split()
index = 0
M = int(data[index])
index += 1
S1 = data[index]
index += 1
S2 = data[index]
index += 1
S3 = data[index]

# Initialize answer
ans = None

# Iterate over all possible residues
for ra in range(M):
    for rb in range(M):
        for rc in range(M):
            if S1[ra] == S2[rb] and S2[rb] == S3[rc]:
                # Compute the frequency of each residue
                res_list = [ra, rb, rc]
                cnt = collections.Counter(res_list)
                # Compute T_config
                T_config = max(res + (cnt[res] - 1) * M for res in cnt)
                # Update ans with minimum T_config
                if ans is None or T_config < ans:
                    ans = T_config

# Output the result
if ans is None:
    print(-1)
else:
    print(ans)