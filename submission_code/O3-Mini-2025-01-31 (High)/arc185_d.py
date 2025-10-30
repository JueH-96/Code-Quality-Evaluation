def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data: 
        return
    try:
        N = int(data[0])
        M = int(data[1])
    except:
        return
    mod = 998244353
    # We have shown that the expected number of operations can be written as:
    #   T = N*M*(M*(N-1) + 1) + [ N*M*(M-1)*( M*(3N-2) + 4 ) ] / 6   .
    # (A short “counting” – or “electrical network” – argument shows that if you sum 
    # over all edges (of which there are n-1 = N*M) the quantity (size(subtree) * size(complement))
    # in our tree then one obtains exactly T.)
    #
    # (For example, when N=2, M=2 one obtains:
    #   Part1 = 2*2*(2*1 + 1) = 4*3 = 12,
    #   Part2 = 2*2*(2-1)*( 2*(3*2-2) + 4 )/6 = 4* ( 2*(6-2)+4 )/6 
    #         = 4* (2*4+4)/6 = 4*12/6 = 8,
    # so that T = 12+8 = 20.)
    
    # To compute division by 6 modulo mod, we compute the modular inverse of 6:
    inv6 = pow(6, mod-2, mod)
    # Part1 = N * M * ( M*(N-1) + 1 )
    part1 = (N % mod) * (M % mod) % mod
    part1 = part1 * ((M*(N-1) + 1) % mod) % mod
    # Part2 = N * M * (M-1) * ( M*(3*N - 2) + 4 ) / 6
    part2 = (N % mod) * (M % mod) % mod
    part2 = part2 * ((M - 1) % mod) % mod
    part2 = part2 * ((M*(3*N - 2) + 4) % mod) % mod
    part2 = part2 * inv6 % mod
    ans = (part1 + part2) % mod
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()