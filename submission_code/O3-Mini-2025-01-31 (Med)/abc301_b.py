def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    seq = list(map(int, data[1:]))

    while True:
        # Check if every adjacent pair has an absolute difference of 1.
        all_adjacent_diff_one = True
        for i in range(len(seq) - 1):
            if abs(seq[i+1] - seq[i]) != 1:
                all_adjacent_diff_one = False
                break
        
        # If they all differ by 1, we can stop
        if all_adjacent_diff_one:
            break

        # Find the first adjacent pair where the absolute difference isn't 1.
        for i in range(len(seq) - 1):
            if abs(seq[i+1] - seq[i]) != 1:
                # Depending on whether the sequence is increasing or decreasing,
                # insert the necessary numbers.
                if seq[i] < seq[i+1]:
                    insertion = list(range(seq[i] + 1, seq[i+1]))
                else:
                    insertion = list(range(seq[i] - 1, seq[i+1], -1))
                seq = seq[:i+1] + insertion + seq[i+1:]
                break

    # Print the final sequence as space-separated integers.
    print(" ".join(map(str, seq)))

if __name__ == '__main__':
    main()