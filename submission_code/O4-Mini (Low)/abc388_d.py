import sys
import threading

def main():
    import sys
    input = sys.stdin.readline
    N = int(input())
    A = list(map(int, input().split()))
    # D is the difference array for future gift receipts
    D = [0] * (N + 3)
    S = 0  # prefix sum of D, gives gifts_received at current i
    B = [0] * N
    for i in range(1, N+1):
        # accumulate gifts_received for alien i
        S += D[i]
        stones_at_i = A[i-1] + S
        # compute how many gifts this alien will give in future
        # they will become adult at i, then give 1 stone at each of the next stones_at_i births,
        # or until year N
        R = i + stones_at_i
        if R > N:
            R = N
        # schedule +1 gift at each year in [i+1, R]
        L = i + 1
        if L <= R:
            D[L] += 1
            D[R+1] -= 1
        # final stones = stones_at_i minus gifts_given = stones_at_i - (R - i)
        B[i-1] = stones_at_i - (R - i)
    # output
    print(" ".join(map(str, B)))

if __name__ == "__main__":
    main()