def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    try:
        V1, V2, V3 = map(int, data[:3])
    except:
        return
    # Check the necessary total-volume condition.
    if V1 + 2 * V2 + 3 * V3 != 1029:
        sys.stdout.write("No")
        return
    found = False
    solX = solY = solZ = 0
    # We will try all possibilities for X, Y, Z in {0,...,7}
    for X in range(8):
        for Y in range(8):
            for Z in range(8):
                if X * Y * Z == V3 and (49 * Y + 7 * X * Z - 2 * X * Y * Z) == V2:
                    solX, solY, solZ = X, Y, Z
                    found = True
                    break
            if found:
                break
        if found:
            break
    if not found:
        sys.stdout.write("No")
        return
    # Set r = 7 - X, s = 7 - Y, t = 7 - Z.
    r = 7 - solX
    s = 7 - solY
    t = 7 - solZ
    # Our chosen configuration then is:
    # Cube1: C(0,0,0)      -> (0 ≤ x,y,z ≤ 7)
    # Cube2: C(0, s, 0)     -> (x from 0 to 7, y from s to s+7, z from 0 to 7)
    # Cube3: C(r, 0, t)     -> (x from r to r+7, y from 0 to 7, z from t to t+7)
    # These cubes satisfy the required intersection volumes.
    sys.stdout.write("Yes
")
    sys.stdout.write(f"0 0 0 0 {s} 0 {r} 0 {t}")
 
if __name__ == '__main__':
    main()