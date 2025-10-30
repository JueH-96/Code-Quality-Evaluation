import sys

def main() -> None:
    sys.setrecursionlimit(1 << 25)           # just in case
    data = sys.stdin.buffer.read().split()
    it = iter(data)

    n = int(next(it))
    deps = [[] for _ in range(n + 1)]        # deps[i] = list of prerequisite books

    for i in range(1, n + 1):
        c = int(next(it))
        deps[i] = [int(next(it)) for _ in range(c)]

    visited = [False] * (n + 1)
    answer = []

    # iterative DFS that pushes a book to `answer`
    # after all its prerequisites have been processed
    stack = [(1, 0)]            # (current_book, next_child_index)
    visited[1] = True

    while stack:
        book, idx = stack[-1]
        if idx < len(deps[book]):            # still have children to process
            nxt = deps[book][idx]
            stack[-1] = (book, idx + 1)      # advance index for this node
            if not visited[nxt]:
                visited[nxt] = True
                stack.append((nxt, 0))
        else:                                # all children done
            stack.pop()
            if book != 1:                    # we never output book 1 itself
                answer.append(book)

    # the produced order already satisfies prerequisite-before-dependent
    sys.stdout.write(' '.join(map(str, answer)))

if __name__ == "__main__":
    main()