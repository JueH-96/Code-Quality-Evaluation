def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    ptr = 0
    n = int(data[ptr])
    ptr += 1
    q = int(data[ptr])
    ptr += 1
    
    A = list(map(int, data[ptr:ptr+n]))
    ptr += n
    
    max_possible = n
    present = [False] * (max_possible + 1)
    for num in A:
        if 0 <= num <= max_possible:
            present[num] = True
    
    output = []
    for _ in range(q):
        i = int(data[ptr]) - 1  # Convert to 0-based index
        ptr += 1
        x = int(data[ptr])
        ptr += 1
        
        old = A[i]
        new = x
        
        if 0 <= old <= max_possible:
            present[old] = False
        
        if 0 <= new <= max_possible:
            present[new] = True
        
        A[i] = new
        
        mex = 0
        while mex <= max_possible and present[mex]:
            mex += 1
        
        if mex <= max_possible:
            output.append(str(mex))
        else:
            output.append(str(max_possible + 1))
    
    print('
'.join(output))

if __name__ == "__main__":
    main()