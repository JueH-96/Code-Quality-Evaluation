import sys

def main() -> None:
    input_data = sys.stdin.read().strip().split()
    it = iter(input_data)
    
    N = int(next(it))
    M = int(next(it))
    S = list(next(it))        # list for easy mutation
    
    # store positions for every colour
    positions_by_colour = [[] for _ in range(M)]
    for idx in range(N):
        c = int(next(it)) - 1          # convert to 0-based colour index
        positions_by_colour[c].append(idx)
    
    # perform the rotations colour by colour
    for pos_list in positions_by_colour:
        k = len(pos_list)
        if k <= 1:
            continue                   # nothing to do
        
        last_char = S[pos_list[-1]]
        # shift elements right
        for j in range(k - 1, 0, -1):
            S[pos_list[j]] = S[pos_list[j - 1]]
        S[pos_list[0]] = last_char
    
    print(''.join(S))

if __name__ == "__main__":
    main()