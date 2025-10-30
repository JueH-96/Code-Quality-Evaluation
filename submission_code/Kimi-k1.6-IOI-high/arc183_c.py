import bisect

MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    
    # Initialize conditions storage (1-based indexing)
    conditions = [[] for _ in range(N + 2)]  # conditions[X] contains tuples (L, R)
    for _ in range(M):
        L = int(input[ptr])
        ptr += 1
        R = int(input[ptr])
        ptr += 1
        X = int(input[ptr])
        ptr += 1
        conditions[X].append((L, R))
    
    occupied = []  # Maintained as a sorted list
    answer = 1
    
    for k in range(N, 0, -1):
        allowed = 0
        allowed_pos = -1
        
        for p in range(1, N + 1):
            if p in occupied:
                continue
            
            # Determine left_large and right_large using binary search
            left_idx = bisect.bisect_right(occupied, p - 1) - 1
            if left_idx >= 0:
                left_large = occupied[left_idx]
            else:
                left_large = -1  # indicates no elements to the left
            
            right_idx = bisect.bisect_left(occupied, p)
            if right_idx < len(occupied):
                right_large = occupied[right_idx]
            else:
                right_large = N + 1  # indicates no elements to the right
            
            active_left = left_large + 1
            active_right = right_large - 1
            
            # Check if this position p is forbidden for element k
            forbidden = False
            for (L, R) in conditions[p]:
                if L >= active_left and R <= active_right:
                    forbidden = True
                    break
            
            if not forbidden:
                allowed += 1
                if allowed_pos == -1:
                    allowed_pos = p  # Keep track of the first allowed position
        
        # Update the answer and occupied list
        if allowed == 0:
            answer = 0
            break
        answer = answer * allowed % MOD
        
        # Insert the chosen position into occupied to maintain sorted order
        bisect.insort(occupied, allowed_pos)
    
    print(answer)

if __name__ == "__main__":
    main()