def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, K = map(int, input_data[:2])
    lost = set(map(int, input_data[2:]))

    # c[i] = how many socks remain of color i+1
    # - If color i+1 is not lost, it has 2 socks
    # - If color i+1 is lost, it has 1 sock
    c = [2] * N
    for x in lost:
        c[x-1] = 1

    # Gather all remaining socks in a sorted list L of their colors.
    # This will have length 2N-K.
    L = []
    for i in range(N):
        if c[i] > 0:
            # color is (i+1), repeated c[i] times
            L.extend([i+1]*c[i])

    # If there's no sock or only one sock, the weirdness is 0 trivially.
    if len(L) < 2:
        print(0)
        return

    # Precompute pair-differences pd[j] = L[j+1] - L[j] for j in [0..len(L)-2]
    pd = [0]*(len(L)-1)
    for j in range(len(L)-1):
        pd[j] = L[j+1] - L[j]

    # Build prefix-cost array:
    # pc[i] = sum of weirdness pairing the first 2*i socks in consecutive pairs.
    # i.e. pc[i] = pd[0] + pd[2] + ... + pd[2*i - 2]
    # pc[0] = 0
    max_pairs = (len(L)//2)  # floor(len(L)/2)
    pc = [0]*(max_pairs+1)
    for i in range(1, max_pairs+1):
        pc[i] = pc[i-1] + pd[2*i - 2]

    # Build suffix-cost array sc[x] = cost of pairing from index x..end-1 in consecutive pairs
    # sc[len(L)] = 0
    # sc[x] = pd[x] + sc[x+2] if x+1 < len(L) else 0
    sc = [0]*(len(L)+1)
    for x in range(len(L)-2, -1, -1):
        if x+1 < len(L):
            if x+2 <= len(L):
                sc[x] = pd[x] + sc[x+2]
            else:
                sc[x] = pd[x]
        # if x+1 >= len(L), sc[x] remains 0 (though that case won't really occur for x < len(L)-1)

    # If we have an even number of socks, we must pair them all:
    if len(L) % 2 == 0:
        print(pc[len(L)//2])
    else:
        # We can skip exactly one sock.  We try skipping each sock i and compute
        # the cost of pairing everything else in consecutive order.
        # cost if we skip index i: pc[i//2] + sc[i+1]
        ans = None
        for i in range(len(L)):
            cost = pc[i//2] + sc[i+1]
            if ans is None or cost < ans:
                ans = cost
        print(ans)

# Let's call solve() to complete the requirement.
def main():
    solve()

if __name__ == "__main__":
    main()