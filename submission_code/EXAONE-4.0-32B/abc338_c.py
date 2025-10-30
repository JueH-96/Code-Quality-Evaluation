def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    Q = list(map(int, data[1:1+n]))
    A = list(map(int, data[1+n:1+2*n]))
    B = list(map(int, data[1+2*n:1+3*n]))
    
    candidates = []
    for i in range(n):
        if A[i] > 0:
            candidates.append(Q[i] // A[i])
    
    if not candidates:
        x_max = 0
    else:
        x_max = min(candidates)
    
    best_total = 0
    for x in range(0, x_max + 1):
        y_max = float('inf')
        for i in range(n):
            remaining = Q[i] - x * A[i]
            if B[i] > 0:
                y_i = remaining // B[i]
                if y_i < y_max:
                    y_max = y_i
        total_servings = x + y_max
        if total_servings > best_total:
            best_total = total_servings
    
    print(best_total)

if __name__ == "__main__":
    main()