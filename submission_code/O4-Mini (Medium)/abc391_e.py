import sys
from array import array

def main():
    data = sys.stdin.read().strip().split()
    if not data:
        return
    # First token is N
    N = int(data[0])
    # Collect all '0'/'1' chars from the rest of the input
    bits = []
    for token in data[1:]:
        # token could be something like '010011101' or '0' etc.
        for ch in token:
            if ch == '0' or ch == '1':
                bits.append(ch)
    m = 3 ** N
    if len(bits) < m:
        # Should not happen in valid input
        raise ValueError("Not enough bits in input")
    # Initialize leaf costs: cost0 = flips to make leaf 0, cost1 = flips to make leaf 1
    # If leaf is '0', cost0=0, cost1=1; if '1', cost0=1, cost1=0
    c0 = array('I', (0 if bits[i] == '0' else 1 for i in range(m)))
    c1 = array('I', (0 if bits[i] == '1' else 1 for i in range(m)))
    # Bottom-up DP: combine triplets until one root remains
    while len(c0) > 1:
        new_len = len(c0) // 3
        nc0 = array('I')  # costs for next level
        nc1 = array('I')
        # Local bindings for speed
        a0 = c0; a1 = c1
        for i in range(0, len(a0), 3):
            pa0 = a0[i];   pa1 = a1[i]
            pb0 = a0[i+1]; pb1 = a1[i+1]
            pc0 = a0[i+2]; pc1 = a1[i+2]
            # Cost to make this node output 0: need at least two children = 0
            # Cases: all three zero, or exactly one child =1
            # all zero:
            cost0 = pa0 + pb0 + pc0
            # one child =1
            tmp = pa1 + pb0 + pc0
            if tmp < cost0: cost0 = tmp
            tmp = pa0 + pb1 + pc0
            if tmp < cost0: cost0 = tmp
            tmp = pa0 + pb0 + pc1
            if tmp < cost0: cost0 = tmp
            # Cost to make this node output 1: need at least two children = 1
            # all three one:
            cost1 = pa1 + pb1 + pc1
            # one child =0
            tmp = pa0 + pb1 + pc1
            if tmp < cost1: cost1 = tmp
            tmp = pa1 + pb0 + pc1
            if tmp < cost1: cost1 = tmp
            tmp = pa1 + pb1 + pc0
            if tmp < cost1: cost1 = tmp
            nc0.append(cost0)
            nc1.append(cost1)
        # Move up one level
        c0, c1 = nc0, nc1

    # At root we have c0[0], c1[0]
    # The original output is the one with cost 0.
    # We want to flip it, so take the other cost.
    root_cost0 = c0[0]
    root_cost1 = c1[0]
    if root_cost0 == 0:
        # original value was 0, to flip to 1 costs root_cost1
        ans = root_cost1
    else:
        # original was 1, to flip to 0 costs root_cost0
        ans = root_cost0
    print(ans)

if __name__ == "__main__":
    main()