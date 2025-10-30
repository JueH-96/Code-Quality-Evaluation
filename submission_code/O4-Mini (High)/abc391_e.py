import sys
from array import array

def main():
    data = sys.stdin.buffer.readline().split()
    if not data:
        return
    N = int(data[0])
    A_line = sys.stdin.buffer.readline().rstrip()
    # Initialize bits array from input string: byte 0/1 per position
    n3 = len(A_line)
    bits = bytearray(n3)
    for i, c in enumerate(A_line):
        # ASCII '0' = 48, '1' = 49, so c&1 gives 0 or 1
        bits[i] = c & 1
    # f array: f[i] = min flips to invert the output of the subtree at leaf i
    # For a leaf, flipping its bit costs 1
    f = array('I', [1]) * n3

    length = n3
    # Bottom-up DP: combine every 3 children into one parent
    for _ in range(N):
        new_len = length // 3
        bits_next = bytearray(new_len)
        f_next = array('I', [0]) * new_len

        b = bits
        fa = f
        # For each new node j, children are at positions 3*j,3*j+1,3*j+2
        for j in range(new_len):
            k = j * 3
            b0 = b[k];    b1 = b[k+1];    b2 = b[k+2]
            f0_k = fa[k]; f1_k = fa[k+1]; f2_k = fa[k+2]
            s = b0 + b1 + b2
            # original output bit of this node
            b_node = 1 if s >= 2 else 0
            # to flip this node, we need the new majority to be 1-b_node
            new_o = 1 - b_node

            # cost to make each child output equal to new_o:
            # zero if child already equals new_o, else cost = f(child)
            if b0 == new_o:
                c0 = 0
            else:
                c0 = f0_k
            if b1 == new_o:
                c1 = 0
            else:
                c1 = f1_k
            if b2 == new_o:
                c2 = 0
            else:
                c2 = f2_k

            # to get majority=new_o, we need at least two children = new_o,
            # so pick the two cheapest to "fix". That's sum(c0,c1,c2) minus the largest one.
            sum_cost = c0 + c1 + c2
            m = c0 if c0 > c1 else c1
            if c2 > m:
                m = c2
            f_node = sum_cost - m

            bits_next[j] = b_node
            f_next[j] = f_node

        bits = bits_next
        f = f_next
        length = new_len

    # After N combines, there's exactly one node; f[0] is the min flips to invert the root
    sys.stdout.write(str(f[0]))


if __name__ == '__main__':
    main()