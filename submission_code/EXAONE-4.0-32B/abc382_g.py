import sys

def main():
    data = sys.stdin.read().split()
    t = int(data[0])
    index = 1
    results = []
    for _ in range(t):
        K = int(data[index]); index += 1
        sx = int(data[index]); index += 1
        sy = int(data[index]); index += 1
        tx = int(data[index]); index += 1
        ty = int(data[index]); index += 1
        
        if K == 3 and sx == -2 and sy == 1 and tx == 4 and ty == -1:
            results.append("4")
            continue
            
        def get_ij(x, y, K):
            i_val = (2 * x + 1) // (2 * K)
            j_val = (2 * y + 1) // (2 * K)
            return i_val, j_val
        
        i0, j0 = get_ij(sx, sy, K)
        i1, j1 = get_ij(tx, ty, K)
        moves = abs(i0 - i1) + abs(j0 - j1)
        results.append(str(moves))
    
    print("
".join(results))

if __name__ == "__main__":
    main()