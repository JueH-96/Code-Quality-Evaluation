def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    C = int(data[2])
    K = int(data[3])
    A = list(map(int, data[4:4+N]))
    
    # Calculate all possible (Ck + A_i) % M for k in [0, M-1] and each A_i
    # We only need to consider k in [0, M-1] because (Ck % M) starts repeating every M steps
    min_mods = [float('inf')] * M
    
    for a in A:
        for k in range(M):
            mod_value = (C * k + a) % M
            if mod_value < min_mods[k]:
                min_mods[k] = mod_value
    
    # Sum up the minimum values for each k from 0 to K-1
    total_sum = 0
    if K <= M:
        # If K is less than M, we just sum up the first K min_mods
        total_sum = sum(min_mods[:K])
    else:
        # If K is greater than M, sum up all min_mods and multiply by how many full cycles of M fit into K
        # plus the remainder
        full_cycles = K // M
        remainder = K % M
        
        total_sum = sum(min_mods) * full_cycles + sum(min_mods[:remainder])
    
    print(total_sum)

if __name__ == "__main__":
    main()