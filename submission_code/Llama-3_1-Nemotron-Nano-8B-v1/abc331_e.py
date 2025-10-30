import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    L = int(input[ptr])
    ptr += 1
    
    a = list(map(int, input[ptr:ptr+N]))
    ptr += N
    b = list(map(int, input[ptr:ptr+M]))
    ptr += M
    
    excluded = set()
    for _ in range(L):
        c = int(input[ptr])
        d = int(input[ptr+1])
        ptr += 2
        excluded.add((c, d))
    
    K = 200  # Adjust K as needed, but 200 should be sufficient
    
    # Prepare top K a's with their indices (1-based)
    a_sorted = sorted([(val, idx+1) for idx, val in enumerate(a)], key=lambda x: (-x[0], x[1]))
    a_top = a_sorted[:K]
    
    # Prepare top K b's with their indices (1-based)
    b_sorted = sorted([(val, idx+1) for idx, val in enumerate(b)], key=lambda x: (-x[0], x[1]))
    b_top = b_sorted[:K]
    
    max_sum = 0
    
    # Check all a in a_top against the first valid b in b_top
    for a_val, a_idx in a_top:
        for b_val, b_idx in b_top:
            if (a_idx, b_idx) not in excluded:
                current_sum = a_val + b_val
                if current_sum > max_sum:
                    max_sum = current_sum
                break  # Break after first valid b for this a
    
    # Check all b in b_top against the first valid a in a_top
    for b_val, b_idx in b_top:
        for a_val, a_idx in a_top:
            if (a_idx, b_idx) not in excluded:
                current_sum = a_val + b_val
                if current_sum > max_sum:
                    max_sum = current_sum
                break  # Break after first valid a for this b
    
    print(max_sum)

if __name__ == '__main__':
    main()