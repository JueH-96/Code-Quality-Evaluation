def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    S = input_data[0]
    Q = int(input_data[1])
    K_list = list(map(int, input_data[2:]))

    # Precompute a table to flip a character's case in O(1).
    # 'A'..'Z' (65..90), 'a'..'z' (97..122).
    flip_table = [None]*128
    for c in range(ord('A'), ord('Z')+1):
        flip_table[c] = chr(c + 32)  # uppercase -> lowercase
    for c in range(ord('a'), ord('z')+1):
        flip_table[c] = chr(c - 32)  # lowercase -> uppercase

    n = len(S)
    max_K = max(K_list)

    # Find the minimum i such that n * 2^i >= max_K
    length = n
    i_max = 0
    while length < max_K:
        length <<= 1
        i_max += 1

    # Prepare answers
    answers = []
    for K in K_list:
        # We find the character in S_i_max (which is large enough to include position K).
        flip_state = 0
        current_length = length
        current_i = i_max
        pos = K  # 1-based index

        while current_i > 0:
            half = current_length >> 1
            if pos > half:
                pos -= half
                flip_state ^= 1
            current_i -= 1
            current_length = half

        # Now current_i == 0, current_length == n
        # pos is between 1 and n (inclusive)
        c = S[pos - 1]
        if flip_state == 1:
            c = flip_table[ord(c)]
        answers.append(c)

    print(" ".join(answers))

# Do not forget to call main()
if __name__ == "__main__":
    main()