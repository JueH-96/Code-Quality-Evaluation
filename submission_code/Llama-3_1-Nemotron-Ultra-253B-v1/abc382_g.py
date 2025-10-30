import sys

def get_tile(K, x, y):
    i = x // K
    j = y // K
    if (i % 2) == (j % 2):
        k = int((y - j * K) // 1)
    else:
        k = int((x - i * K) // 1)
    return (i, j, k)

def main():
    input = sys.stdin.read().split()
    T = int(input[0])
    idx = 1
    for _ in range(T):
        K = int(input[idx])
        Sx = int(input[idx+1])
        Sy = int(input[idx+2])
        Tx = int(input[idx+3])
        Ty = int(input[idx+4])
        idx +=5
        
        x_start = Sx + 0.5
        y_start = Sy + 0.5
        x_end = Tx + 0.5
        y_end = Ty + 0.5
        
        i1, j1, k1 = get_tile(K, x_start, y_start)
        i2, j2, k2 = get_tile(K, x_end, y_end)
        
        manhattan = abs(i1 - i2) + abs(j1 - j2)
        dk = abs(k1 - k2)
        print(manhattan + dk)

if __name__ == '__main__':
    main()