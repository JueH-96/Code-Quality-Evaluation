def is_arithmetic(seq):
    if len(seq) <= 2:
        return True
    diff = seq[1] - seq[0]
    for i in range(2, len(seq)):
        if seq[i] - seq[i-1] != diff:
            return False
    return True

def find_arithmetic_subsequences(arr, k, MOD):
    n = len(arr)
    count = 0
    
    def generate_subsequences(pos, curr_len, curr_seq):
        nonlocal count
        if curr_len == k:
            if is_arithmetic(curr_seq):
                count = (count + 1) % MOD
            return
        
        if pos >= n or n - pos < k - curr_len:
            return
            
        # Include current element
        generate_subsequences(pos + 1, curr_len + 1, curr_seq + [arr[pos]])
        # Skip current element
        generate_subsequences(pos + 1, curr_len, curr_seq)
    
    generate_subsequences(0, 0, [])
    return count

N = int(input())
A = list(map(int, input().split()))
MOD = 998244353

result = []
for k in range(1, N+1):
    result.append(str(find_arithmetic_subsequences(A, k, MOD)))

print(" ".join(result))