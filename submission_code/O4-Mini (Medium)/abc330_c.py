import sys
import threading

def main():
    import sys
    from math import isqrt
    
    data = sys.stdin.read().strip()
    if not data:
        return
    D = int(data)
    best = D  # initialize with |0+0 - D| = D
    
    x = 0
    # We'll scan x from 0 upward, break when further x can't improve best
    while True:
        xsq = x*x
        if xsq > D:
            # best possible with y=0 is xsq - D
            diff = xsq - D
            if diff >= best:
                # further x only increases xsq, so diff only grows
                break
            best = diff
        else:
            # we need y^2 close to (D - x^2)
            rem = D - xsq
            y0 = isqrt(rem)
            # try y0
            diff0 = rem - y0*y0
            if diff0 < 0:
                diff0 = -diff0
            if diff0 < best:
                best = diff0
            # try y0+1
            y1 = y0 + 1
            diff1 = ys1 = y1*y1
            diff1 = xsq + diff1 - D
            if diff1 < 0:
                diff1 = -diff1
            if diff1 < best:
                best = diff1
            
            # if best is zero, we found perfect match
            if best == 0:
                break
        
        x += 1

    print(best)

if __name__ == "__main__":
    main()