def main():
    import sys
    input = sys.stdin.readline

    T = int(input())
    SHIFT = 10**6

    out = []
    for _ in range(T):
        R,B = map(int,input().split())
        N = R + B
        # Quick impossibility check:
        if (R % 2 == 1) or (R==0 and B%2==1):
            out.append("No")
            continue
        # Otherwise we can build:
        out.append("Yes")
        pts = []   # (piece_char, x, y)
        x0 = SHIFT
        y0 = SHIFT
        # Case R=0, B even:
        if R==0:
            # diag-only cycle of length B
            # pattern NE^1, NW^(B/2-1), SW^1, SE^(B/2-1)
            k = B//2
            a = 1
            b = k-1
            # start
            x,y = x0,y0
            pts.append(('B',x,y))
            # 1 NE
            for _ in range(a):
                x+=1; y+=1
                pts.append(('B',x,y))
            # b NW
            for _ in range(b):
                x-=1; y+=1
                pts.append(('B',x,y))
            # 1 SW
            x-=1; y-=1
            pts.append(('B',x,y))
            # b SE
            for _ in range(b):
                x+=1; y-=1
                pts.append(('B',x,y))
            # we have made B steps => B+1 points, last repeats first
            # drop the last repeated:
            pts.pop()
        # Case B=0, R even:
        elif B==0:
            # orth-only cycle of length R
            # pattern E^1, N^(R/2-1), W^1, S^(R/2-1)
            k = R//2
            a = 1
            b = k-1
            x,y = x0,y0
            pts.append(('R',x,y))
            # 1 E
            for _ in range(a):
                x+=1
                pts.append(('R',x,y))
            # b N
            for _ in range(b):
                y+=1
                pts.append(('R',x,y))
            # 1 W
            x-=1
            pts.append(('R',x,y))
            # b S
            for _ in range(b):
                y-=1
                pts.append(('R',x,y))
            pts.pop()
        else:
            # R>0, R even, B>0 (split off the general 4-block interleaving)
            r2 = R//2
            b2 = B//2
            # start
            x,y = x0,y0
            pts.append(('R',x,y))
            # r2 times East
            for _ in range(r2):
                x+=1
                pts.append(('R',x,y))
            # b2 times NE
            for _ in range(b2):
                x+=1; y+=1
                pts.append(('B',x,y))
            # r2 times West
            for _ in range(r2):
                x-=1
                pts.append(('R',x,y))
            # b2 times SW
            for _ in range(b2):
                x-=1; y-=1
                pts.append(('B',x,y))
            # that is R+B steps, back to the start
            # drop the last repeated
            pts.pop()

        # now emit the points
        for c,xx,yy in pts:
            out.append(f"{c} {xx} {yy}")

    sys.stdout.write("
".join(out))

if __name__=="__main__":
    main()