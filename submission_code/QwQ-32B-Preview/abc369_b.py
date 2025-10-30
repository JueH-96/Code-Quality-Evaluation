def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    operations = []
    index = 1
    for _ in range(N):
        A_i = int(data[index])
        S_i = data[index + 1]
        operations.append((A_i, S_i))
        index += 2
    
    # Separate L and R sequences
    L_seq = [A for A, S in operations if S == 'L']
    R_seq = [A for A, S in operations if S == 'R']
    
    # Calculate sum of distances for L sequence
    if L_seq:
        L_sum = sum(abs(L_seq[i] - L_seq[i-1]) for i in range(1, len(L_seq)))
    else:
        L_sum = 0
    
    # Calculate sum of distances for R sequence
    if R_seq:
        R_sum = sum(abs(R_seq[i] - R_seq[i-1]) for i in range(1, len(R_seq)))
    else:
        R_sum = 0
    
    # Total minimal fatigue is the sum of both sums
    total_fatigue = L_sum + R_sum
    print(total_fatigue)

if __name__ == '__main__':
    main()