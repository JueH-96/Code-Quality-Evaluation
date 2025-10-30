N = int(input())
S = input().strip()

total = 0
for k in range(N):  # 0-indexed
    digit = int(S[k])
    # Number of substrings starting at or before position k
    count = k + 1
    # Sum of powers of 10: 1 + 10 + 10^2 + ... + 10^(N-k-1)
    if N - k - 1 == 0:
        power_sum = 1
    else:
        power_sum = (pow(10, N - k) - 1) // 9
    
    contribution = digit * count * power_sum
    total += contribution

print(total)