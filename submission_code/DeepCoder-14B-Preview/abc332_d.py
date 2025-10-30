import itertools

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    H = int(input[ptr])
    ptr += 1
    W = int(input[ptr])
    ptr += 1

    A = []
    for _ in range(H):
        row = list(map(int, input[ptr:ptr+W]))
        ptr += W
        A.append(row)
    
    B = []
    for _ in range(H):
        row = list(map(int, input[ptr:ptr+W]))
        ptr += W
        B.append(row)
    
    min_operations = None

    # Generate all possible row permutations and their inversion counts
    for P in itertools.permutations(range(H)):
        inv_p = 0
        for i in range(len(P)):
            for j in range(i+1, len(P)):
                if P[i] > P[j]:
                    inv_p += 1
        
        # Generate all possible column permutations and their inversion counts
        for Q in itertools.permutations(range(W)):
            inv_q = 0
            for i in range(len(Q)):
                for j in range(i+1, len(Q)):
                    if Q[i] > Q[j]:
                        inv_q += 1
            
            # Build the transformed grid
            transformed = []
            for i in range(H):
                new_row = [A[P[i]][q] for q in Q]
                transformed.append(new_row)
            
            # Check if the transformed grid matches B
            if transformed == B:
                total = inv_p + inv_q
                if min_operations is None or total < min_operations:
                    min_operations = total
    
    if min_operations is not None:
        print(min_operations)
    else:
        print(-1)

if __name__ == "__main__":
    main()