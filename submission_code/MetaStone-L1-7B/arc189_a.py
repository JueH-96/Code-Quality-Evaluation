MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    A0 = [(i % 2) for i in range(1, N+1)]
    
    # Function to check if a block can be formed by a merge operation
    def is_valid_block(A0, l, r):
        if A0[l] != A0[r]:
            return False
        val = A0[l]
        for i in range(l+1, r):
            if A0[i] == val:
                return False
        return True
    
    # Find all blocks in the target array
    blocks = []
    if not A:
        return 1
    current = A[0]
    start = 0
    for i in range(1, N):
        if A[i] != current:
            blocks.append((start, i))
            current = A[i]
            start = i
    blocks.append((start, N))
    
    total = 1
    for (l, r) in blocks:
        # Find the left and right positions in the initial array that form this block
        # Find the first occurrence in A0 where A0[i] == current
        # and check if it's within [l, r]
        # But this is tricky because the initial A0 may not match the block's value
        # So we need to find all positions in A0 that could form this block
        
        # The block in A is formed by merging a range in A0 that starts and ends with the block's value
        # and all cells in between are different from the block's value
        
        # So for the block in A, the corresponding positions in A0 must be such that the block's value is the same as A0[l] and A0[r]
        # but since the block's value may be different from A0[l] and A0[r], we need to find the positions in A0 that match
        
        # However, this is complicated, so perhaps we can model it as follows:
        # Each block in A corresponds to a certain interval in A0 that starts and ends with the block's value
        # and all cells in between are different from the block's value
        
        # So for each block, find all possible (l0, r0) in A0 where A0[l0] == A0[r0] == block_val, and all cells in between are different
        
        # But since the block in A is formed by a single merge operation, the corresponding interval in A0 must be such that the block's value is the same as A0[l0] and A0[r0], and all cells in between are different
        
        # Thus, for each block in A, the number of valid intervals is the number of possible (l0, r0) in A0 that satisfy the above conditions
        
        # To compute this, we can iterate over all possible l0 and r0 in A0 where A0[l0] == A0[r0] == block_val, and check if all cells in between are different
        
        block_val = current
        count = 0
        n = len(A0)
        for l0 in range(n):
            if A0[l0] != block_val:
                continue
            for r0 in range(l0 + 1, n):
                if A0[r0] != block_val:
                    continue
                valid = True
                for i in range(l0 + 1, r0):
                    if A0[i] == block_val:
                        valid = False
                        break
                if valid:
                    count += 1
        total = (total * count) % MOD
    print(total)

if __name__ == "__main__":
    main()