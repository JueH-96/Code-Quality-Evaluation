N = int(input())
A = list(map(int, input().split()))

# Compute prefix XORs
prefix = [0] * (N + 1)
for i in range(N):
    prefix[i + 1] = prefix[i] ^ A[i]

total = 0

# For each bit position
for bit in range(30):  # 30 bits should be enough
    # Precompute prefix counts for this bit
    count0 = [0] * (N + 2)
    count1 = [0] * (N + 2)
    
    for j in range(N + 1):
        count0[j + 1] = count0[j]
        count1[j + 1] = count1[j]
        if (prefix[j] >> bit) & 1:
            count1[j + 1] += 1
        else:
            count0[j + 1] += 1
    
    contribution = 0
    for i in range(N - 1):  # i from 0 to N-2
        if (prefix[i] >> bit) & 1:
            # prefix[i] has bit = 1, count k ∈ [i+2, N] with bit = 0
            contribution += count0[N + 1] - count0[i + 2]
        else:
            # prefix[i] has bit = 0, count k ∈ [i+2, N] with bit = 1
            contribution += count1[N + 1] - count1[i + 2]
    
    total += contribution * (1 << bit)

print(total)