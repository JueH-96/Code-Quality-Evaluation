import sys
import threading

def main():
    import sys

    data = sys.stdin.read().split()
    N, K = map(int, data[:2])
    S = data[2]

    blocks = []
    i = 0
    # find all 1-blocks
    while i < N:
        if S[i] == '1':
            j = i
            while j < N and S[j] == '1':
                j += 1
            # block from i to j-1
            blocks.append((i, j - 1))
            i = j
        else:
            i += 1

    # zero-based indices:
    l_prev, r_prev = blocks[K-2]
    l_k, r_k = blocks[K-1]
    len_k = r_k - l_k + 1

    # build the result
    prefix = S[:r_prev+1]
    ones_block = '1' * len_k
    zeros_count = r_k - (r_prev + len_k)
    zeros_block = '0' * zeros_count
    suffix = S[r_k+1:]

    result = prefix + ones_block + zeros_block + suffix
    sys.stdout.write(result)

if __name__ == "__main__":
    main()