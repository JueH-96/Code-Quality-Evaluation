import math

def get_tile_info(x, y, K):
    x += 0.5
    y += 0.5
    i = int(x // K)
    # Check if same parity
    # i and j must have same parity for x to be in [i*K, (i+1)*K)
    # For same parity, j must be such that j*K + k <= y < j*K + k + 1
    # For different parity, i*K +k <= x < i*K +k +1 and j*K <= y < (j+1)*K
    # Determine parity of i and j
    # First, check if x is in same parity tile
    # Same parity tile: i and j have same parity
    # Different parity tile: i and j have different parity
    # Compute i for same parity case
    i_same = int((x - 0.5) / K)
    # Check if x is in [i_same*K, (i_same+1)*K)
    if i_same*K <= x < (i_same+1)*K:
        # Same parity
        j = int((y - 0.5) // K)
        k = (y - 0.5) - j*K
        if k < 0:
            j -= 1
            k += K
        if (i_same % 2) == (j % 2):
            return (i_same, j, int(k))
    # If not same parity, compute for different parity
    i_diff = int(x // K)
    # x must be in [i_diff*K +k, i_diff*K +k +1)
    # So k = floor(x - i_diff*K)
    k = int(x - i_diff*K)
    if k < 0:
        i_diff -= 1
        k += K
    if i_diff*K + k <= x < i_diff*K + k +1:
        j = int((y - 0.5) // K)
        if (i_diff % 2) != (j % 2):
            return (i_diff, j, k)
    # If none, fallback (should not happen)
    return (i, int((y - 0.5) // K), int((y - 0.5) % K))

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    T = int(input[idx])
    idx +=1
    for _ in range(T):
        K = int(input[idx])
        Sx = int(input[idx+1])
        Sy = int(input[idx+2])
        Tx = int(input[idx+3])
        Ty = int(input[idx+4])
        idx +=5
        
        # Get start tile info
        sx_center = Sx + 0.5
        sy_center = Sy + 0.5
        i1, j1, k1 = get_tile_info(sx_center, sy_center, K)
        
        # Get target tile info
        tx_center = Tx + 0.5
        ty_center = Ty + 0.5
        i2, j2, k2 = get_tile_info(tx_center, ty_center, K)
        
        # Compute Manhattan distance between (i1,j1) and (i2,j2)
        dx = abs(i1 - i2)
        dy = abs(j1 - j2)
        manhattan = dx + dy
        
        # Check parity of start and end
        start_parity = (i1 + j1) % 2
        end_parity = (i2 + j2) % 2
        
        k_diff = abs(k1 - k2)
        
        if start_parity == end_parity:
            manhattan += k_diff * 2
        else:
            manhattan += k_diff
        
        print(manhattan)

if __name__ == '__main__':
    main()