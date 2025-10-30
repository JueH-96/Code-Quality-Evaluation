import sys

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    # Precompute sum_mask using dynamic programming
    sum_mask = [0] * (1 << N)
    for mask in range(1, 1 << N):
        lb = mask & -mask
        idx = (lb).bit_length() - 1
        sum_mask[mask] = A[idx] + sum_mask[mask ^ lb]
    
    memo = {}
    
    def dfs(mask):
        if mask == 0:
            return {0}
        if mask in memo:
            return memo[mask]
        
        first_bit = mask & -mask
        fe = (first_bit).bit_length() - 1
        first_bit = 1 << fe
        rest_mask = mask ^ first_bit
        
        res = set()
        s_rest = 0
        while True:
            if (s_rest & rest_mask) == s_rest:
                s = first_bit | s_rest
                remaining_mask = mask ^ s
                inner_set = dfs(remaining_mask)
                sum_s = sum_mask[s]
                for x in inner_set:
                    res.add(sum_s ^ x)
            if s_rest == rest_mask:
                break
            s_rest += 1
        
        memo[mask] = res
        return res
    
    full_mask = (1 << N) - 1
    result_set = dfs(full_mask)
    print(len(result_set))

if __name__ == "__main__":
    main()