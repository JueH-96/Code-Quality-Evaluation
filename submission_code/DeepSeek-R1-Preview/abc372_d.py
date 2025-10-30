def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    H = list(map(int, input[1:N+1]))
    
    # Compute PGE for each index (0-based)
    stack = []
    pge = [ -1 ] * N  # -1 means no PGE
    for j in range(N):
        while stack and H[stack[-1]] < H[j]:
            stack.pop()
        if stack:
            pge[j] = stack[-1]
        else:
            pge[j] = -1  # No PGE
        stack.append(j)
    
    # Initialize difference array
    diff = [0] * (N + 2)  # 1-based to N
    
    for j in range(N):
        k = pge[j]
        a = k + 1  # convert to 0-based
        b = j - 1
        # convert a and b to 1-based
        a_1 = a + 1
        b_1 = b + 1
        if a_1 <= b_1:
            diff[a_1] += 1
            if b_1 + 1 <= N:
                diff[b_1 + 1] -= 1
    
    # Compute prefix sum
    res = [0] * (N + 1)  # 1-based
    for i in range(1, N + 1):
        res[i] = res[i - 1] + diff[i]
    
    # Now, handle the case where j is directly after i (i = j-1)
    for j in range(1, N):
        i = j  # j in 0-based is i+1 in 1-based
        res[i + 1] += 1  # Check if this is needed
    
    output = res[1:N+1]
    for i in range(N):
        output[i] = res[i + 1]
    
    # Now, subtract the cases where i = j-1 is already counted
    # But this part is tricky and requires more careful handling
    
    # Print the result
    print(' '.join(map(str, output[:N])))

if __name__ == '__main__':
    main()