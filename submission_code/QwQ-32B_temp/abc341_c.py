import numpy as np

def main():
    import sys
    H, W, N = map(int, sys.stdin.readline().split())
    T = sys.stdin.readline().strip()
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    
    # Create land array: land[i][j] is True if it's land ('.')
    land = np.array([[c == '.' for c in row] for row in grid], dtype=bool)
    prev = land.copy()
    
    for d in T:
        current = np.zeros((H, W), dtype=bool)
        if d == 'L':
            current[:,:-1] = land[:,:-1] & prev[:,1:]
        elif d == 'R':
            current[:,1:] = land[:,1:] & prev[:,:-1]
        elif d == 'U':
            current[:-1, :] = land[:-1, :] & prev[1:,:]
        elif d == 'D':
            current[1:,:] = land[1:,:] & prev[:-1, :]
        prev = current.copy()
    
    print(np.sum(prev))

if __name__ == "__main__":
    main()