# YOUR CODE HERE
import sys

def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    N_K = sys.stdin.readline().strip()
    while N_K == '':
        N_K = sys.stdin.readline().strip()
    N, K = map(int, N_K.split())
    S = sys.stdin.read().strip()
    while len(S) < N:
        S += sys.stdin.read().strip()
    S = list(S[:N])

    blocks = []
    i = 0
    while i < N:
        if S[i] == '1' and (i == 0 or S[i-1] == '0'):
            l = i +1
            # Find the end of the block
            j = i
            while j +1 < N and S[j+1] == '1':
                j +=1
            r = j +1
            blocks.append((l, r))
            i = j +1
        else:
            i +=1

    # K >=2 and len(blocks) >=K
    prev_block = blocks[K-2]
    current_block = blocks[K-1]

    r_prev = prev_block[1] -1  # 0-based
    l_k = current_block[0] -1  # 0-based
    r_k = current_block[1] -1  # 0-based
    len_k = r_k - l_k +1

    # Set S[r_prev +1 : r_prev + len_k +1] = '1's
    # Set S[r_prev + len_k +1 : r_k +1] = '0's
    # Ensure indices are within bounds
    # 'r_prev +1' to 'r_prev + len_k' inclusive
    start_ones = r_prev +1
    end_ones = r_prev + len_k +1
    if start_ones < N:
        end_ones = min(end_ones, N)
        S[start_ones:end_ones] = ['1'] * (end_ones - start_ones)

    # 'r_prev + len_k +1' to 'r_k' inclusive
    start_zeros = r_prev + len_k +1
    end_zeros = r_k +1
    if start_zeros < N and start_zeros < end_zeros:
        S[start_zeros:end_zeros] = ['0'] * (end_zeros - start_zeros)

    print(''.join(S))

if __name__ == "__main__":
    main()