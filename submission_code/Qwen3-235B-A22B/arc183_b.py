import sys
import bisect
from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    results = []
    
    for _ in range(T):
        N, K = int(input[ptr]), int(input[ptr+1])
        ptr +=2
        A = list(map(int, input[ptr:ptr+N]))
        ptr += N
        B = list(map(int, input[ptr:ptr+N]))
        ptr += N
        
        # Step 1: Check if all elements in B are present in A
        set_A = set(A)
        for val in B:
            if val not in set_A:
                results.append("No")
                break
        else:
            # Step 2: Check for each position i in B if B[i] exists in its window
            pos_dict = defaultdict(list)
            for idx in range(N):
                val = B[idx]
                pos_dict[val].append(idx + 1)  # 1-based index
            
            possible = True
            for i in range(N):
                current_val = B[i]
                left = (i+1) - K  # 1-based
                right = (i+1) + K
                left = max(1, left)
                right = min(N, right)
                
                lst = pos_dict.get(current_val, [])
                if not lst:
                    possible = False
                    break
                # Find if any element in lst is within [left, right]
                pos = bisect.bisect_left(lst, left)
                if pos < len(lst) and lst[pos] <= right:
                    continue
                else:
                    possible = False
                    break
            if possible:
                results.append("Yes")
            else:
                results.append("No")
    
    sys.stdout.write('
'.join(results) + '
')

if __name__ == "__main__":
    main()