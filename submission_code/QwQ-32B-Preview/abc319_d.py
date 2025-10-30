def is_feasible(W, L, M):
    line_count = 1
    current_width = 0
    for width in L:
        if current_width + width + (1 if current_width > 0 else 0) > W:
            line_count += 1
            current_width = width
            if line_count > M:
                return False
        else:
            current_width += width + (1 if current_width > 0 else 0)
    return True

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    L = list(map(int, data[2:2+N]))
    
    W_min = max(L)
    W_max = sum(L) + (N - 1)
    
    low = W_min
    high = W_max
    
    while low < high:
        mid = (low + high) // 2
        if is_feasible(mid, L, M):
            high = mid
        else:
            low = mid + 1
    
    print(low)

if __name__ == "__main__":
    main()