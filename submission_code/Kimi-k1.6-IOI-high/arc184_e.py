MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr +=1
    M = int(input[ptr])
    ptr +=1
    
    # Preprocess all sequences and group by the first element
    groups = dict()
    for _ in range(N):
        seq = list(map(int, input[ptr:ptr+M]))
        ptr += M
        key = seq[0]
        if key not in groups:
            groups[key] = []
        groups[key].append(seq[1:])  # Store the rest elements
    
    total = 0
    
    for key in groups:
        lst = groups[key]
        K = len(lst)
        # Compute the sum of f(i,j) for all i <= j in lst
        cnt = 0
        for i in range(K):
            for j in range(i, K):
                a = lst[i]
                b = lst[j]
                # Compute XOR of a and b
                xor = 0
                for k in range(len(a)):
                    if a[k] != b[k]:
                        xor |= 1 << (len(a) - 1 - k)
                # Now compute the minimal x for this xor vector
                x = 0
                current_xor = xor
                for bit in range(len(a)-1, -1, -1):
                    mask = 1 << bit
                    if (current_xor & mask) != 0:
                        if (x & mask) == 0:
                            x ^= mask
                            # Flip all lower bits
                            lower_mask = mask -1
                            current_xor ^= lower_mask
                total += x
                total %= MOD
    print(total % MOD)
    
if __name__ == "__main__":
    main()