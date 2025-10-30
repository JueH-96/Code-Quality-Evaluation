import sys

def main():
    data = sys.stdin.readline().strip().split()
    if not data:
        return
    N = int(data[0])

    # Minimum number of bits (friends) needed to distinguish N bottles:
    # We need M bits so that 2^M >= N.
    # If we label bottles 1..N by codes 0..N-1, then M = (N-1).bit_length()
    M = (N - 1).bit_length()

    # Output M
    print(M, flush=True)

    # For each friend i (bit position i), serve all bottles whose code (j-1)
    # has the i-th bit set.
    for i in range(M):
        subset = []
        for j in range(1, N+1):
            # code = j-1
            if ((j-1) >> i) & 1:
                subset.append(j)
        # Print the size and the actual bottle numbers
        if subset:
            print(len(subset), *subset, flush=True)
        else:
            # If no bottles have this bit, still print "0"
            print(0, flush=True)

    # Read the response string S of length M
    S = sys.stdin.readline().strip()
    # Reconstruct the spoiled bottle's code from the OR-responses
    code = 0
    # S[i] corresponds to bit i
    for i, ch in enumerate(S):
        if ch == '1':
            code |= (1 << i)
    # Our bottles were labeled by code+1
    answer = code + 1

    # Print the answer and exit
    print(answer, flush=True)

if __name__ == "__main__":
    main()