def main():
    import sys
    Sx, Sy = map(int, sys.stdin.readline().split())
    Tx, Ty = map(int, sys.stdin.readline().split())
    
    def same_tile(sx, sy, tx, ty):
        if (sx, sy) == (tx, ty):
            return True
        if sy == ty and abs(sx - tx) == 1:
            return (sx + sy) % 2 == 0
        if sx == tx and abs(sy - ty) == 1:
            return (sx + sy) % 2 == 1
        return False
    
    if same_tile(Sx, Sy, Tx, Ty):
        print(0)
    else:
        dx = abs(Sx - Tx)
        dy = abs(Sy - Ty)
        print(max(dx, dy))

if __name__ == "__main__":
    main()