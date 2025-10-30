import sys

def solve():
    N, K = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()

    blocks = []
    i = 0
    while i < N:
        if S[i] == '0':
            i += 1
            continue
        else: # S[i] == '1'
            block_start = i
            # Advance i to find the end of this block of 1s
            while i < N and S[i] == '1':
                i += 1
            block_end = i - 1 
            blocks.append((block_start, block_end))
    
    # K is 1-based index for blocks in the problem statement.
    # We need the (K-1)-th block and K-th block.
    # In our 0-indexed `blocks` list, these correspond to elements at indices K-2 and K-1.
    
    # (K-1)-th block is B_prev (previous block relative to K-th block)
    # Its 0-based indices are (l_prev, r_prev)
    l_prev, r_prev = blocks[K-2] 
    
    # K-th block is B_curr (current block to be moved)
    # Its 0-based indices are (l_curr, r_curr)
    l_curr, r_curr = blocks[K-1]
    
    # We construct the new string T by rearranging parts of S.
    # The logic is: (Prefix up to end of B_prev) + (B_curr) + (Zeros between B_prev and B_curr) + (Suffix after B_curr)
    # These parts are:
    # Part A: S[0 ... r_prev]
    # Part C (B_curr): S[l_curr ... r_curr]
    # Part B (zeros): S[r_prev + 1 ... l_curr - 1]
    # Part D (suffix): S[r_curr + 1 ... N-1]
    
    # Using Python string slicing (0-based, end index exclusive):
    s_A = S[0 : r_prev + 1]
    s_C = S[l_curr : r_curr + 1]
    s_B = S[r_prev + 1 : l_curr]
    s_D = S[r_curr + 1 : N]
    
    # Concatenate the parts in the new order.
    # Using "".join() is generally efficient for concatenating multiple string parts.
    result_parts = [s_A, s_C, s_B, s_D]
    result = "".join(result_parts)
    
    print(result)

if __name__ == '__main__':
    solve()