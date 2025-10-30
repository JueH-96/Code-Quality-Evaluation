def main() -> None:
    import sys

    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    P = list(map(int, data[1:]))

    # pair each score with its original index
    indexed_scores = [(score, idx) for idx, score in enumerate(P)]
    # sort by score in descending order
    indexed_scores.sort(key=lambda x: -x[0])

    ranks = [0] * N               # final answer
    current_rank = 1              # rank to assign to current group
    i = 0
    while i < N:
        # find the extent of the group with the same score
        j = i
        score_val = indexed_scores[i][0]
        while j < N and indexed_scores[j][0] == score_val:
            j += 1
        group_size = j - i

        # assign current_rank to everyone in this group
        for k in range(i, j):
            original_idx = indexed_scores[k][1]
            ranks[original_idx] = current_rank

        # advance
        current_rank += group_size
        i = j

    # output
    out_lines = '
'.join(map(str, ranks))
    sys.stdout.write(out_lines + '
')

# call main
if __name__ == "__main__":
    main()