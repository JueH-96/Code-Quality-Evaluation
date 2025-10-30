def main():
    import sys
    Sx, Sy, Tx, Ty = map(int, sys.stdin.readline().split())
    # Check if start and target are the same tile
    def same_tile(Sx, Sy, Tx, Ty):
        if (Sx + Sy) % 2 != (Tx + Ty) % 2:
            return False
        if (Sx + Sy) % 2 == 0:
            # Both horizontal
            return Sy == Ty and (Sx == Tx or Sx + 1 == Tx)
        else:
            # Both vertical
            return Sx == Tx and (Sy == Ty or Sy + 1 == Ty)
    
    if same_tile(Sx, Sy, Tx, Ty):
        print(0)
        return
    
    # Transform coordinates
    def transform(x, y):
        # For horizontal tiles (even sum)
        if (x + y) % 2 == 0:
            u = x + y
            v = x - y
        else:
            # Vertical tiles (odd sum)
            u = x + y + 1
            v = x - y
        return (u, v)
    
    u1, v1 = transform(Sx, Sy)
    u2, v2 = transform(Tx, Ty)
    
    du = abs(u1 - u2)
    dv = abs(v1 - v2)
    
    ans = (du + dv) // 2
    print(ans)

if __name__ == '__main__':
    main()