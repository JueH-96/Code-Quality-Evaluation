# YOUR CODE HERE
import sys
import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    
    W = int(input[0])
    H = int(input[1])
    N = int(input[2])
    
    strawberries = []
    idx = 3
    for _ in range(N):
        p = int(input[idx])
        q = int(input[idx + 1])
        strawberries.append((p, q))
        idx += 2
    
    A = int(input[idx])
    idx += 1
    a = list(map(int, input[idx:idx + A]))
    idx += A
    
    B = int(input[idx])
    idx += 1
    b = list(map(int, input[idx:idx + B]))
    
    a = [0] + a + [W]
    b = [0] + b + [H]
    
    # Count strawberries in each piece
    piece_count = [[0] * (B + 1) for _ in range(A + 1)]
    
    for p, q in strawberries:
        x_idx = bisect.bisect_left(a, p) - 1
        y_idx = bisect.bisect_left(b, q) - 1
        piece_count[x_idx][y_idx] += 1
    
    # Find min and max strawberries in any piece
    min_strawberries = float('inf')
    max_strawberries = float('-inf')
    
    for i in range(A):
        for j in range(B):
            min_strawberries = min(min_strawberries, piece_count[i][j])
            max_strawberries = max(max_strawberries, piece_count[i][j])
    
    print(min_strawberries, max_strawberries)

if __name__ == "__main__":
    main()