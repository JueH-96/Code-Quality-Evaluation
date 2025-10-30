import sys

def main():
    N, K = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()

    blocks = []
    current_block = None
    for i in range(N):
        if S[i] == '1':
            if current_block is None:
                current_block = [i, i]
            else:
                current_block[1] = i
        else:
            if current_block is not None:
                blocks.append((current_block[0], current_block[1]))
                current_block = None
    if current_block is not None:
        blocks.append((current_block[0], current_block[1]))

    K_prev = K - 2
    K_block = K - 1
    block_prev = blocks[K_prev]
    block_K = blocks[K_block]

    l_prev, r_prev = block_prev
    l_K, r_K = block_K
    length_K = r_K - l_K + 1

    part1 = S[:r_prev + 1]
    part2 = '1' * length_K

    start_zero = r_prev + length_K + 1
    end_zero = r_K
    zeros_length = end_zero - start_zero + 1
    part3 = '0' * zeros_length

    part4 = S[r_K + 1:]

    new_S = part1 + part2 + part3 + part4
    print(new_S)

if __name__ == "__main__":
    main()