def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return

    N = int(data[0])
    S = data[1]

    # For each character, we want to know the maximum length when it appears in a contiguous segment.
    # That maximum length corresponds to distinct substrings "c", "cc",... "c" * (max_count).
    max_counts = {}

    # Traverse the string S and compute maximum contiguous counts per character.
    if N > 0:
        current_char = S[0]
        current_length = 1

        for i in range(1, N):
            if S[i] == current_char:
                current_length += 1
            else:
                # update maximum for the current character
                if current_char in max_counts:
                    max_counts[current_char] = max(max_counts[current_char], current_length)
                else:
                    max_counts[current_char] = current_length
                # reset for the next character segment
                current_char = S[i]
                current_length = 1
        # update for the last segment in the string
        if current_char in max_counts:
            max_counts[current_char] = max(max_counts[current_char], current_length)
        else:
            max_counts[current_char] = current_length

    # The total number of valid one-character repeated substrings is the sum of maximum segment lengths
    # for each letter, because for a maximum segment of length m, the distinct substrings are:
    # c, cc, ..., c * m.
    total_substrings = sum(max_counts[ch] for ch in max_counts)
    
    sys.stdout.write(str(total_substrings))

if __name__ == '__main__':
    main()