import sys
import threading

def main():
    import sys
    from math import isqrt

    data = sys.stdin.read().strip().split()
    if not data:
        return
    R = int(data[0])
    R2 = R * R

    total = 0
    # We sweep over i >= 0.  For each square centered at (±i, ±j),
    # the farthest corner from the origin is at (|i|+0.5,|j|+0.5).
    # The condition (|i|+0.5)^2 + (|j|+0.5)^2 <= R^2
    # rewrites to (2i+1)^2 + (2j+1)^2 <= 4*R^2.  Fix i, let u=2i+1, S=4R^2 - u^2.
    # We need (2j+1)^2 <= S, so 2j+1 <= floor(sqrt(S)) = t, thus j <= (t-1)//2.
    #
    # After finding j_max=(t-1)//2 >= 0, we count how many (i,j) pairs in all quadrants:
    #   if i=0 and j=0: 1 square
    #   if i=0, j>0 or i>0, j=0: 2 squares each
    #   if i>0, j>0: 4 squares each.
    #
    # Summing by cases:
    #   i==0: weight = 1 for j=0, then 2 for each j=1..j_max => 1 + 2*j_max
    #   i>0: weight = 2 for j=0, then 4 for each j=1..j_max => 2 + 4*j_max

    for i in range(0, R+1):
        u = 2*i + 1
        S = 4*R2 - u*u
        if S < 1:
            break
        t = isqrt(S)
        if t < 1:
            break
        j_max = (t - 1) // 2
        if j_max < 0:
            continue
        if i == 0:
            total += 1 + 2 * j_max
        else:
            total += 2 + 4 * j_max

    print(total)


if __name__ == "__main__":
    main()