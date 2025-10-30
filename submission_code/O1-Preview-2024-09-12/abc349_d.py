# YOUR CODE HERE
import sys
import math

def main():
    import sys
    input = sys.stdin.readline
    L_str, R_str = sys.stdin.read().split()
    L = int(L_str)
    R = int(R_str)
    curr = L
    intervals = []
    while curr < R:
        if curr == 0:
            exponent = int(math.floor(math.log2(R - curr)))
        else:
            exponent = (curr & -curr).bit_length() -1
        while True:
            if exponent < 0:
                length = 1
                break
            length = 1 << exponent
            if curr % length == 0 and length <= R - curr:
                break
            exponent -=1
        intervals.append( (curr, curr + length) )
        curr += length
    print(len(intervals))
    for l, r in intervals:
        print(f"{l} {r}")

if __name__ == "__main__":
    main()