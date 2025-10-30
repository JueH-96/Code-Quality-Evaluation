import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    N, K = map(int, data[:2])
    S = data[2].strip()
    # Find the intervals of 1-blocks
    blocks = []
    i = 0
    while i < N:
        if S[i] == '1' and (i == 0 or S[i-1] == '0'):
            # start of a block
            l = i
            j = i
            while j < N and S[j] == '1':
                j += 1
            r = j - 1
            blocks.append((l, r))
            i = j
        else:
            i += 1
        if len(blocks) >= K:
            # we only need up to the K-th block
            break

    # Extract the needed indices (0-based)
    prev_l, prev_r = blocks[K-2]  # (K-1)-th block
    cur_l,  cur_r  = blocks[K-1]  # K-th block
    length = cur_r - cur_l + 1

    # Build result as a list for in-place edits
    T = list(S)

    # 1) positions up to prev_r stay the same
    # 2) from prev_r+1 to prev_r+length: set to '1'
    for idx in range(prev_r+1, prev_r+1 + length):
        T[idx] = '1'
    # 3) from prev_r+1+length to cur_r: set to '0'
    for idx in range(prev_r+1 + length, cur_r+1):
        T[idx] = '0'
    # 4) after cur_r everything stays the same

    # Output
    sys.stdout.write(''.join(T))

if __name__ == "__main__":
    main()