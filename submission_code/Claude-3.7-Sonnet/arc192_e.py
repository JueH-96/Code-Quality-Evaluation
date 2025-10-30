# YOUR CODE HERE
def solve(W, H, L, R, D, U):
    MOD = 998244353
    
    def is_block(x, y):
        return (0 <= x <= W and 0 <= y <= H and (x < L or x > R or y < D or y > U))
    
    # Start with row 0
    prev_row = [0] * (W + 1)
    
    total = 0
    
    for x in range(W + 1):
        if is_block(x, 0):
            prev_row[x] = 1
            if x > 0 and is_block(x - 1, 0):
                prev_row[x] = (prev_row[x] + prev_row[x - 1]) % MOD
            total = (total + prev_row[x]) % MOD
    
    # Process subsequent rows
    for y in range(1, H + 1):
        curr_row = [0] * (W + 1)
        
        for x in range(W + 1):
            if not is_block(x, y):
                continue
            
            # Path of length 0 starting at (x, y)
            curr_row[x] = 1
            
            # Path coming from the left
            if x > 0 and is_block(x - 1, y):
                curr_row[x] = (curr_row[x] + curr_row[x - 1]) % MOD
            
            # Path coming from below
            if is_block(x, y - 1):
                curr_row[x] = (curr_row[x] + prev_row[x]) % MOD
            
            total = (total + curr_row[x]) % MOD
        
        prev_row = curr_row
    
    return total

if __name__ == "__main__":
    W, H, L, R, D, U = map(int, input().split())
    print(solve(W, H, L, R, D, U))