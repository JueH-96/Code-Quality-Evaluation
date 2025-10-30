# YOUR CODE HERE
K = int(input())
C = list(map(int, input().split()))
MOD = 998244353

dp = {}
def count_strings(length, counts):
    if length == 0:
        return 1
    if (length, tuple(counts)) in dp:
        return dp[(length, tuple(counts))]
    
    total_count = 0
    for i in range(26):
        if counts[i] > 0:
            new_counts = list(counts)
            new_counts[i] -= 1
            total_count = (total_count + count_strings(length - 1, new_counts)) % MOD
    
    dp[(length, tuple(counts))] = total_count
    return total_count

total_strings = 0
for length in range(1, K + 1):
    total_strings = (total_strings + count_strings(length, C)) % MOD

print(total_strings)