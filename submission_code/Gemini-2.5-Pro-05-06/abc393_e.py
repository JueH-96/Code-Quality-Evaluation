import sys

def solve():
    line1 = sys.stdin.readline().split()
    N = int(line1[0])
    K = int(line1[1])
    
    A_str_list = sys.stdin.readline().split()
    A = [int(x) for x in A_str_list]

    max_val_in_a = 0
    if N > 0:
        for x_val in A:
            if x_val > max_val_in_a:
                max_val_in_a = x_val
    else: 
        return 

    freq = [0] * (max_val_in_a + 1)
    pos = [[] for _ in range(max_val_in_a + 1)]

    for i in range(N):
        val = A[i]
        freq[val] += 1
        pos[val].append(i)

    counts = [0] * (max_val_in_a + 1)
    for g_divisor in range(1, max_val_in_a + 1):
        for multiple_of_g in range(g_divisor, max_val_in_a + 1, g_divisor):
            counts[g_divisor] += freq[multiple_of_g]

    ans = [0] * N 
    for g_candidate in range(max_val_in_a, 0, -1):
        if counts[g_candidate] >= K:
            for multiple_val in range(g_candidate, max_val_in_a + 1, g_candidate):
                if freq[multiple_val] > 0: 
                    for original_idx in pos[multiple_val]:
                        if ans[original_idx] == 0:
                            ans[original_idx] = g_candidate
    
    for i in range(N):
        sys.stdout.write(str(ans[i]) + "
")

solve()