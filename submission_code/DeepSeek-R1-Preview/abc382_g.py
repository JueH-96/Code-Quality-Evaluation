import math

def get_tile_coords(x, y, K):
    i = math.floor(x / K)
    j = math.floor(y / K)
    rem_x = x - i * K
    rem_y = y - j * K
    if (i % 2) == (j % 2):
        # Same parity: horizontal strip, k is based on y
        k = math.floor(rem_y)
    else:
        # Different parity: vertical strip, k is based on x
        k = math.floor(rem_x)
    return (i, j, k)

def transformed_coords(i, j, k, K):
    if (i % 2) == (j % 2):
        tx = i
        ty = j * K + k
    else:
        tx = i * K + k
        ty = j
    return (tx, ty)

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    T = int(input[idx])
    idx +=1
    for _ in range(T):
        K = int(input[idx])
        Sx = int(input[idx+1])
        Sy = int(input[idx+2])
        Tx = int(input[idx+3])
        Ty = int(input[idx+4])
        idx +=5
        
        start_x = Sx + 0.5
        start_y = Sy + 0.5
        end_x = Tx + 0.5
        end_y = Ty + 0.5
        
        i1, j1, k1 = get_tile_coords(start_x, start_y, K)
        i2, j2, k2 = get_tile_coords(end_x, end_y, K)
        
        tx1, ty1 = transformed_coords(i1, j1, k1, K)
        tx2, ty2 = transformed_coords(i2, j2, k2, K)
        
        distance = abs(tx1 - tx2) + abs(ty1 - ty2)
        print(distance)

if __name__ == "__main__":
    main()