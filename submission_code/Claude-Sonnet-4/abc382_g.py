def find_tile(x, y, K):
    i = x // K
    j = y // K
    
    if x < 0 and x % K != 0:
        i -= 1
    if y < 0 and y % K != 0:
        j -= 1
    
    rel_x = x - i * K
    rel_y = y - j * K
    
    if (i + j) % 2 == 0:
        k = rel_y
    else:
        k = rel_x
    
    return (i, j, k)

def min_moves(K, sx, sy, tx, ty):
    start = find_tile(sx, sy, K)
    target = find_tile(tx, ty, K)
    
    if start == target:
        return 0
    
    si, sj, sk = start
    ti, tj, tk = target
    
    # If same (i,j), just need to change k
    if si == ti and sj == tj:
        return 1
    
    # Calculate minimum moves
    di = abs(ti - si)
    dj = abs(tj - sj)
    
    # Base moves to get to the right (i,j)
    base_moves = di + dj
    
    # Check if we need extra moves for k coordination
    if (si + sj) % 2 == (ti + tj) % 2:
        # Same parity pattern
        if sk == tk:
            return base_moves
        else:
            return base_moves + 1
    else:
        # Different parity pattern
        return base_moves + 1

T = int(input())
for _ in range(T):
    K, sx, sy, tx, ty = map(int, input().split())
    result = min_moves(K, sx, sy, tx, ty)
    print(result)