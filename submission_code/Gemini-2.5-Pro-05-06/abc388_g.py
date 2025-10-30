import sys

def main():
    N = int(sys.stdin.readline())
    A_list = list(map(int, sys.stdin.readline().split())) # Renamed to avoid conflict with A as char
    Q = int(sys.stdin.readline())

    output_buffer = []

    for _ in range(Q):
        L_str, R_str = sys.stdin.readline().split()
        L, R = int(L_str), int(R_str)
        
        # Adjust to 0-indexed
        L_idx, R_idx = L - 1, R - 1
        
        num_mochi_in_query = R_idx - L_idx + 1

        if num_mochi_in_query < 2: # Need at least 2 mochi to form a pair
            output_buffer.append("0")
            continue

        # Binary search for K, the number of pairs
        # Max K is num_mochi_in_query // 2
        low_k = 0
        high_k = num_mochi_in_query // 2
        ans_k = 0 # Stores the maximum K found so far that works

        while low_k <= high_k:
            # Current candidate for K
            mid_k = low_k + (high_k - low_k) // 2 
            
            if mid_k == 0: # 0 pairs is always possible
                # ans_k will be at least 0, this just ensures we try mid_k=1 next
                ans_k = max(ans_k, mid_k) 
                low_k = mid_k + 1
                continue

            # Check if mid_k pairs can be formed
            # Tops: A_list[L_idx ... L_idx + mid_k - 1]
            # Bottoms: A_list[R_idx - mid_k + 1 ... R_idx]
            # The s-th smallest top (0-indexed s) is A_list[L_idx + s]
            # The s-th smallest of chosen bottoms is A_list[R_idx - mid_k + 1 + s]
            
            possible_this_k = True
            for s in range(mid_k):
                top_mochi_val = A_list[L_idx + s]
                bottom_mochi_val = A_list[R_idx - mid_k + 1 + s]
                
                if 2 * top_mochi_val > bottom_mochi_val:
                    possible_this_k = False
                    break
            
            if possible_this_k:
                ans_k = mid_k # mid_k pairs are possible, try for more
                low_k = mid_k + 1
            else:
                # mid_k pairs not possible, try fewer
                high_k = mid_k - 1
        
        output_buffer.append(str(ans_k))

    sys.stdout.write("
".join(output_buffer) + "
")

if __name__ == '__main__':
    main()