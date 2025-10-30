import sys

def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    # Initial string S
    S = data[0].decode()
    n = len(S)
    # Number of queries
    Q = int(data[1])
    # Query positions (as byte-strings)
    Ks = data[2:2+Q]

    # Prepare result list
    res = []
    # Use built-in bit_count for popcount
    bc = int.bit_count

    for x in Ks:
        K = int(x)
        # Zero‐based block index and offset within the block
        b, offset = divmod(K-1, n)
        # Parity of pops in block index decides flip
        if bc(b) & 1:
            # flip case
            res.append(S[offset].swapcase())
        else:
            res.append(S[offset])

    # Output answers separated by spaces
    sys.stdout.write(" ".join(res))

if __name__ == "__main__":
    main()