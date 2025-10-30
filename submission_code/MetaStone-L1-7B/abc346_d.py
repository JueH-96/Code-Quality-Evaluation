def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    S = data[1]
    C = list(map(int, data[2:2+N]))
    
    a0 = [0] * N
    a1 = [0] * N
    
    for j in range(N):
        # Compute a0[j] and a1[j]
        current = int(S[j])
        desired0 = 0 if j % 2 == 0 else 1
        a0[j] = 0 if current == desired0 else C[j]
        
        desired1 = 1 if j % 2 == 0 else 0
        a1[j] = 0 if current == desired1 else C[j]
    
    # Compute prefix sums for a0 and a1
    prefix_a0 = [0] * (N)
    prefix_a1 = [0] * (N)
    prefix_a0[0] = a0[0]
    prefix_a1[0] = a1[0]
    for j in range(1, N):
        prefix_a0[j] = prefix_a0[j-1] + a0[j]
        prefix_a1[j] = prefix_a1[j-1] + a1[j]
    
    # Compute suffix sums for a0 and a1
    suffix_a0 = [0] * (N)
    suffix_a1 = [0] * (N)
    suffix_a0[-1] = a0[-1]
    suffix_a1[-1] = a1[-1]
    for j in range(N-2, -1, -1):
        suffix_a0[j] = suffix_a0[j+1] + a0[j]
        suffix_a1[j] = suffix_a1[j+1] + a1[j]
    
    min_cost = float('inf')
    
    for i in range(N-1):
        # Case A: first part ends with 0
        caseA = prefix_a0[i] + suffix_a0[i+1]
        # Case B: first part ends with 1
        caseB = prefix_a1[i] + suffix_a1[i+1]
        current_min = min(caseA, caseB)
        if current_min < min_cost:
            min_cost = current_min
    
    print(min_cost)

if __name__ == '__main__':
    main()