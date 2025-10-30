def main():
    import sys

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1]

    # Array to keep track of the maximum run length for each character
    max_run = [0]*26

    current_char = S[0]
    current_count = 1
    max_run[ord(current_char) - ord('a')] = 1

    for i in range(1, N):
        if S[i] == current_char:
            current_count += 1
        else:
            # Update max run for the character we just finished
            idx = ord(current_char) - ord('a')
            if current_count > max_run[idx]:
                max_run[idx] = current_count

            # Start a new run
            current_char = S[i]
            current_count = 1

        # Update max run on the fly
        idx_new = ord(S[i]) - ord('a')
        if current_count > max_run[idx_new]:
            max_run[idx_new] = current_count

    print(sum(max_run))

# Do not forget to call main()
main()