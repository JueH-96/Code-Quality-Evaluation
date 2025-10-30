import sys
from array import array

def main() -> None:
    # read whole input, split by any whitespace
    tokens = sys.stdin.buffer.read().split()
    N = int(tokens[0])
    # concatenate the remaining tokens (they may be split by spaces/newlines)
    bits_bytes = b''.join(tokens[1:])

    # build initial arrays
    bits   = bytearray()      # current value (0/1) of each node on this level
    cost0  = array('I')       # min flips inside the subtree to force this node to 0
    cost1  = array('I')       # min flips inside the subtree to force this node to 1
    for b in bits_bytes:
        if b == 49:           # '1'
            bits.append(1)
            cost0.append(1)   # need one flip to make it 0
            cost1.append(0)   # already 1
        else:                 # '0'
            bits.append(0)
            cost0.append(0)   # already 0
            cost1.append(1)   # need one flip to make it 1

    length = len(bits)        # current number of nodes on this level

    # bottom-up DP through the 3-ary tree
    while length > 1:
        new_bits  = bytearray()
        new_cost0 = array('I')
        new_cost1 = array('I')

        i = 0
        while i < length:
            # children's current values
            b0, b1, b2 = bits[i], bits[i+1], bits[i+2]
            # majority value for the parent
            new_bits.append(1 if b0 + b1 + b2 >= 2 else 0)

            # children's costs
            c00, c01, c02 = cost0[i],   cost0[i+1],   cost0[i+2]
            c10, c11, c12 = cost1[i],   cost1[i+1],   cost1[i+2]

            # best cost to make parent output 0  (need at least two 0's)
            best0 = c00 + c01 + c02                         # 000
            tmp   = c00 + c01 + c12                         # 001
            if tmp < best0: best0 = tmp
            tmp   = c00 + c11 + c02                         # 010
            if tmp < best0: best0 = tmp
            tmp   = c10 + c01 + c02                         # 100
            if tmp < best0: best0 = tmp

            # best cost to make parent output 1  (need at least two 1's)
            best1 = c10 + c11 + c12                         # 111
            tmp   = c10 + c11 + c02                         # 110
            if tmp < best1: best1 = tmp
            tmp   = c10 + c01 + c12                         # 101
            if tmp < best1: best1 = tmp
            tmp   = c00 + c11 + c12                         # 011
            if tmp < best1: best1 = tmp

            new_cost0.append(best0)
            new_cost1.append(best1)

            i += 3

        # move up one level
        bits, cost0, cost1 = new_bits, new_cost0, new_cost1
        length //= 3

    # root information
    root_val = bits[0]
    answer   = cost1[0] if root_val == 0 else cost0[0]
    print(answer)

if __name__ == "__main__":
    main()