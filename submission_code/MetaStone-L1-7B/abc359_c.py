import sys

def main():
    Sx, Sy = map(int, sys.stdin.readline().split())
    Tx, Ty = map(int, sys.stdin.readline().split())
    
    dx = abs(Tx - Sx)
    dy = abs(Ty - Sy)
    
    same_tile_steps = min(dx, dy)
    tiles = dx + dy - same_tile_steps
    
    print(tiles)

if __name__ == '__main__':
    main()