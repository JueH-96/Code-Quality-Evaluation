def count_valid_pairs(N, M, A):
    prefix_mod = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix_mod[i] = (prefix_mod[i - 1] + A[i - 1]) % M
    
    mod_count = {}
    for mod in prefix_mod:
        if mod in mod_count:
            mod_count[mod] += 1
        else:
            mod_count[mod] = 1
    
    result = 0
    for count in mod_count.values():
        if count > 1:
            result += count * (count - 1) // 2
    
    return result

# Reading input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:]))

# Output the result
print(count_valid_pairs(N, M, A))