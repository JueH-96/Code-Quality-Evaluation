import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    K = int(input[ptr])
    ptr += 1
    A = []
    B = []
    for _ in range(N):
        a = int(input[ptr])
        ptr += 1
        b = int(input[ptr])
        ptr += 1
        A.append(a)
        B.append(b)
    
    if K == 0:
        print(0)
        return
    
    # Step 1: Initialize prev_top with the first step's top K values
    prev_top = []
    if K >= 1:
        candidates = [(A[i] + B[i], i) for i in range(N)]
        candidates.sort(reverse=True)
        prev_top = candidates[:K]
    
    # Process steps from 2 to K
    for step in range(2, K + 1):
        current_candidates = []
        for i in range(N):
            max_prev_val = -1
            # Check all entries in prev_top to find the maximum value that does not use function i
            for (val, func_idx) in prev_top:
                if func_idx != i:
                    if val > max_prev_val:
                        max_prev_val = val
            current_val = max_prev_val * A[i] + B[i]
            current_candidates.append((current_val, i))
        # Sort and keep top K candidates
        current_candidates.sort(reverse=True)
        prev_top = current_candidates[:K]
    
    print(prev_top[0][0])

if __name__ == "__main__":
    main()