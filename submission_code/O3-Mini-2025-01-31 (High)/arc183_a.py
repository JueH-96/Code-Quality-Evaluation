# YOUR CODE HERE
import sys, math

def main():
    data = sys.stdin.read().strip().split()
    if not data:
        return
    try:
        N = int(data[0])
        K = int(data[1])
    except:
        return
        
    # Special case: if N == 1, then the only good sequence is K copies of 1.
    if N == 1:
        sys.stdout.write(" ".join(["1"] * K))
        return

    # For N>=2 the total number S = (N*K)!/(K!)^N is even, so the median index X = S/2.
    # Relative rank r = X/S = 1/2.
    r_num, r_den = 1, 2  # r = 1/2 exactly

    counts = [K] * N
    T = N * K  # total number of positions remaining

    res = []

    # While there are positions to fill...
    while T > 0:
        for i in range(N):
            if counts[i] == 0:
                continue
            # p = counts[i] / T
            p_num = counts[i]
            p_den = T
            # We want to check if r <= p.
            # That is, check if: r_num/r_den <= p_num/p_den <=> r_num * p_den <= p_num * r_den.
            if r_num * p_den <= p_num * r_den:
                # Candidate (i+1) is chosen.
                # Update r = r / p = (r_num * p_den) / (r_den * p_num)
                new_r_num = r_num * p_den
                new_r_den = r_den * p_num
                g = math.gcd(new_r_num, new_r_den)
                r_num, r_den = new_r_num // g, new_r_den // g

                res.append(i + 1)
                counts[i] -= 1
                T -= 1
                break
            else:
                # Otherwise skip candidate i's block: update r = r - p.
                # r = (r_num/r_den) - (p_num/p_den)
                #   = (r_num * p_den - p_num * r_den) / (r_den * p_den)
                r_num = r_num * p_den - p_num * r_den
                r_den = r_den * p_den
                g = math.gcd(r_num, r_den)
                r_num //= g
                r_den //= g
    sys.stdout.write(" ".join(map(str, res)) + "
")

if __name__ == '__main__':
    main()