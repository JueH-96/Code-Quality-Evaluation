import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    S = int(data[1])
    A = list(map(int, data[2:]))

    sumN = sum(A)
    # If S is a multiple of full cycle sum and at least one cycle
    if sumN != 0 and S % sumN == 0 and S >= sumN:
        print("Yes")
        return

    # Otherwise look for remainder rem = S mod sumN (or S if S < sumN)
    rem = S % sumN
    # if rem == 0 but S < sumN, then rem==S%sumN==S, positive; only rem==0 when S is multiple
    # and we already handled full-cycle case, so rem > 0 here.
    if rem == 0:
        # This only happens if S < sumN == 0 or something degenerate; 
        # but A_i > 0 so sumN>0, so rem==0 only if S is multiple which we handled.
        rem = sumN

    # We need to see if some contiguous subarray in the infinite periodic array sums to rem.
    # It's enough to check in a window of length up to N over A+A.
    B = A + A
    l = 0
    curr = 0
    # sliding window on B
    for r in range(len(B)):
        curr += B[r]
        # shrink while too large or window too long
        while l <= r and (curr > rem or r - l + 1 > N):
            curr -= B[l]
            l += 1
        if curr == rem:
            print("Yes")
            return
    print("No")

if __name__ == "__main__":
    main()