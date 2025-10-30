def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    ptr = 0
    N = int(data[ptr])
    ptr +=1
    Q = int(data[ptr])
    ptr +=1
    A = list(map(int, data[ptr:ptr+N]))
    ptr +=N
    
    # Initialize frequency array for 0..N
    freq = [0]*(N+2)  # 0 to N, and N+1 as a dummy
    for num in A:
        if num <= N:
            freq[num] +=1
    
    # Build segment tree
    size = 1
    while size < (N+1):
        size <<=1
    tree = [N+1]*(2*size)
    
    for m in range(N+1):
        if freq[m] ==0:
            tree[size + m] = m
        else:
            tree[size + m] = N+1
    # Fill remaining leaves with N+1
    for m in range(N+1, size):
        tree[size + m] = N+1
    
    # Build the tree
    for i in range(size-1, 0, -1):
        tree[i] = min(tree[2*i], tree[2*i+1])
    
    # Process queries
    results = []
    for _ in range(Q):
        i = int(data[ptr])-1  # convert to 0-based
        ptr +=1
        x = int(data[ptr])
        ptr +=1
        
        old_val = A[i]
        new_val = x
        
        # Update old_val
        if old_val <= N:
            freq[old_val] -=1
            if freq[old_val] ==0:
                pos = size + old_val
                tree[pos] = old_val
                pos >>=1
                while pos >=1:
                    tree[pos] = min(tree[2*pos], tree[2*pos+1])
                    pos >>=1
        
        # Update new_val
        if new_val <= N:
            freq[new_val] +=1
            if freq[new_val] ==1:
                pos = size + new_val
                tree[pos] = N+1
                pos >>=1
                while pos >=1:
                    tree[pos] = min(tree[2*pos], tree[2*pos+1])
                    pos >>=1
        
        A[i] = new_val
        results.append(str(tree[1]))
    
    print('
'.join(results))

if __name__ == '__main__':
    main()