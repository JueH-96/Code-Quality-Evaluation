import sys

def solve():
    N, T = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    row_counts = [0] * N
    col_counts = [0] * N
    diag1_count = 0  # Main diagonal: r == c
    diag2_count = 0  # Anti-diagonal: r + c == N - 1

    for turn_idx in range(T):  # turn_idx is 0-indexed (0 to T-1)
        num_announced = A[turn_idx]
        
        # Convert num_announced to 0-indexed (r, c)
        r = (num_announced - 1) // N
        c = (num_announced - 1) % N
        
        # Update row count
        row_counts[r] += 1
        if row_counts[r] == N:
            print(turn_idx + 1)  # turn_idx + 1 is the 1-indexed turn number
            return
            
        # Update col count
        col_counts[c] += 1
        if col_counts[c] == N:
            print(turn_idx + 1)
            return
            
        # Update main diagonal count
        if r == c:
            diag1_count += 1
            if diag1_count == N:
                print(turn_idx + 1)
                return
        
        # Update anti-diagonal count
        if r + c == N - 1:
            diag2_count += 1
            if diag2_count == N:
                print(turn_idx + 1)
                return
                
    # If loop finishes, Bingo was not achieved
    print("-1")

if __name__ == '__main__':
    solve()