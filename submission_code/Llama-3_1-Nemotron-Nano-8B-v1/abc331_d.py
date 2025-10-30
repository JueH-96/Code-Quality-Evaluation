import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    
    b_indices = []
    for _ in range(N):
        line = input[ptr].strip()
        ptr += 1
        b_indices.append([j for j, c in enumerate(line) if c == 'B'])
    
    for _ in range(Q):
        A = int(input[ptr])
        B = int(input[ptr+1])
        C = int(input[ptr+2])
        D = int(input[ptr+3])
        ptr +=4
        
        # Compute R_i for each residue i
        R = [( (C - i) // N - ( (A-1 - i) // N ) ) for i in range(N)]
        
        # Compute C_j for each residue j
        C_j = [ ( (D - j) // N - ( (B-1 - j) // N ) ) for j in range(N) ]
        
        total = 0
        for i in range(N):
            total += R[i] * sum(C_j[j] for j in b_indices[i])
        
        print(total)

if __name__ == '__main__':
    main()