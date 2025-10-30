import sys

def main():
    n = int(sys.stdin.readline())
    P = list(map(int, sys.stdin.readline().split()))
    Q = list(map(int, sys.stdin.readline().split()))
    
    posP = [0] * (n + 1)  # 1-based indexing for rows
    posQ = [0] * (n + 1)  # 1-based indexing for columns
    
    # Fill posP according to the permutation P
    for idx in range(n):
        row = P[idx]
        posP[row] = idx + 1  # position in permutation (1-based)
    
    # Fill posQ according to the permutation Q
    for idx in range(n):
        col = Q[idx]
        posQ[col] = idx + 1
    
    # Generate the grid
    for i in range(1, n+1):
        r_i = (n + 1) - posP[i]  # reverse the order for row i
        row = []
        for j in range(1, n+1):
            c_j = posQ[j]
            if r_i > c_j:
                row.append('0')
            else:
                row.append('1')
        print(''.join(row))
    
if __name__ == '__main__':
    main()