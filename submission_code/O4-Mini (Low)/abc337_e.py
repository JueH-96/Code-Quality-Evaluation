import sys
import math

def main():
    input = sys.stdin.readline
    # Read N
    N_line = input().strip()
    if not N_line:
        return
    N = int(N_line)
    # Compute minimum number of friends = number of bits needed
    M = math.ceil(math.log2(N))
    # Edge case: if N == 1, no bits needed, but constraints N>=2 so skip
    
    # Prepare which bottles each friend (bit position) will test
    # We'll number bits from 0 (LSB) to M-1 (MSB)
    groups = []
    for bit in range(M):
        group = []
        for bottle in range(1, N+1):
            # If the (bottle-1)'s bit-th bit is 1, assign to this friend
            if ((bottle-1) >> bit) & 1:
                group.append(bottle)
        groups.append(group)
    
    # Output M
    print(M, flush=True)
    # Output each group's size and the bottles
    for group in groups:
        # Ki and list
        print(len(group), *group, flush=True)
    
    # Read the upset string S
    S = input().strip()
    # Reconstruct the spoiled index-1 by bits
    idx = 0
    for bit, ch in enumerate(S):
        if ch == '1':
            idx |= (1 << bit)
    # Convert back to 1-based bottle number
    spoiled = idx + 1
    # Print the answer
    print(spoiled, flush=True)

if __name__ == "__main__":
    main()