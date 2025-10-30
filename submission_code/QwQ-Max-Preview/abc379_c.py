import sys

def main():
    n, m = map(int, sys.stdin.readline().split())
    x = list(map(int, sys.stdin.readline().split()))
    a = list(map(int, sys.stdin.readline().split()))
    
    total_stones = sum(a)
    if total_stones != n:
        print(-1)
        return
    
    # Sort the cells by their positions
    cells = sorted(zip(x, a), key=lambda p: p[0])
    x_sorted = [p[0] for p in cells]
    a_sorted = [p[1] for p in cells]
    
    previous_end = 0
    total_ops = 0
    possible = True
    
    for i in range(m):
        current_x = x_sorted[i]
        current_a = a_sorted[i]
        current_start = max(current_x, previous_end + 1)
        current_end = current_start + current_a - 1
        
        if current_end > n:
            possible = False
            break
        
        # Calculate the operations needed for this block
        steps = current_a * (current_start - current_x) + (current_a * (current_a - 1)) // 2
        total_ops += steps
        
        previous_end = current_end
    
    # Check if the last block ends exactly at n (sum of a is n, so previous_end should be n)
    if possible and previous_end == n:
        print(total_ops)
    else:
        print(-1)

if __name__ == "__main__":
    main()