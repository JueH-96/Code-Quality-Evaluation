import sys

def main():
    import sys
    from collections import defaultdict
    data = sys.stdin.read().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    Q = int(data[ptr])
    ptr += 1
    A = [0] * (N + 1)
    freq = defaultdict(int)
    for i in range(1, N + 1):
        A[i] = int(data[ptr])
        freq[A[i]] += 1
        ptr += 1
    
    # Find initial mex
    current_mex = 0
    while freq[current_mex] > 0:
        current_mex += 1
    
    # Process each query
    for _ in range(Q):
        i_k = int(data[ptr])
        ptr += 1
        x_k = int(data[ptr])
        ptr += 1
        old_value = A[i_k]
        freq[old_value] -= 1
        if old_value < current_mex and freq[old_value] == 0:
            current_mex = old_value
        A[i_k] = x_k
        if x_k < current_mex and freq[x_k] == 0:
            current_mex = x_k
        freq[x_k] += 1
        if x_k == current_mex:
            while freq[current_mex] > 0:
                current_mex += 1
        print(current_mex)

if __name__ == "__main__":
    main()