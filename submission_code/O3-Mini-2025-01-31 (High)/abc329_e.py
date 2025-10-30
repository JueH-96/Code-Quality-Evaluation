def main():
    import sys
    from collections import deque
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    M = int(data[1])
    S = data[2].strip()
    T = data[3].strip()
    
    # We will simulate the reverse process.
    # Think of it this way: if it is possible to form S (starting from all '#' and stamping T),
    # then there is a way (in reverse) to "unstamp" S to all '#' (like erasing the letters).
    # In the reverse process, we imagine that a window (of length M) can be "erased" if for
    # every position in that window either the letter equals the corresponding letter of T 
    # or it has already been erased (turned to '#' in our simulation). To kick‐start the process,
    # we identify all windows where S[i:i+M] already exactly equals T. (In our simulation we use
    # sets: "made" for positions that match and "todo" for positions that do not.)
    
    # windows[i] = [made, todo] for the window starting at index i.
    windows = []
    for i in range(N - M + 1):
        made = set()
        todo = set()
        for j in range(M):
            if S[i+j] == T[j]:
                made.add(i+j)
            else:
                todo.add(i+j)
        windows.append([made, todo])
    
    # 'stamped' will mark the indices that’ve been “erased” (i.e. turned to '#')
    stamped = [False] * N
    q = deque()
    # Initially add all windows that are immediately unstampable (i.e. no conflict; todo is empty)
    for i in range(len(windows)):
        if not windows[i][1]:
            q.append(i)
    
    # For an index j, the windows covering j are those with starting positions from
    # max(0, j-M+1) to min(j, N-M) (inclusive) because a window starting at i covers indices i..i+M-1.
    while q:
        i = q.popleft()
        # For window i, positions from i to i+M-1 were stamped in the forward process.
        for j in range(i, i + M):
            if not stamped[j]:
                stamped[j] = True
                # For every window covering index j, update its todo set.
                start = max(0, j - M + 1)
                end = min(N - M, j)
                for k in range(start, end + 1):
                    if j in windows[k][1]:
                        windows[k][1].remove(j)
                        if not windows[k][1]:
                            q.append(k)
    
    # If every position has been stamped (i.e. erased) in the reverse process, then it is possible.
    sys.stdout.write("Yes" if all(stamped) else "No")

if __name__ == '__main__':
    main()