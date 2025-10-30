def main():
    import sys
    from collections import deque

    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    # S: target string we wish to obtain (of length n)
    S = list(next(it).strip())
    # T: the stamp string we can write (of length m)
    T = list(next(it).strip())

    N = n  # length of S
    M = m

    # We will simulate the reverse process.
    # In the forward process we start with a string X = "####..." (all '#' characters)
    # and then repeatedly stamp T (i.e. replace M consecutive characters with T).
    # Reverse idea: start from S and “unstamp” portions that could have been produced
    # in the last stamping. In reverse, a valid stamping interval i (0-indexed) is one whose
    # window (positions i to i+M-1) has all characters either matching T (the letter that would
    # have been stamped) or already been “removed” (replaced with '#'). Initially none are removed.
    #
    # For each possible window (i = 0 to N-M) we record two sets:
    #  made: indices j in window where S[j] already equals T[j-i]
    #  todo: indices j in window where S[j] does not equal T[j-i] (these positions must be "unstamped"
    #        later to be able to stamp here).
    windows = []  # Will store a list of [made, todo] for each window starting at index i.
    # For each position in S, track all window indices covering that position.
    pos = [[] for _ in range(N)]
    
    for i in range(N - M + 1):
        made = set()
        todo = set()
        for j in range(M):
            if S[i+j] == T[j]:
                made.add(i+j)
            else:
                todo.add(i+j)
        windows.append([made, todo])
        for j in range(i, i+M):
            pos[j].append(i)
    
    # The "done" array indicates positions that have been "unstamped" (i.e. we treat that letter as '#')
    done = [False] * N
    doneCount = 0

    # Queue for windows that become stampable in reverse. When a window has an empty 'todo' set,
    # it means we can imagine it was stamped in the final forward process.
    q = deque()
    
    # Initially, any window whose todo is empty can be removed.
    for i in range(len(windows)):
        if not windows[i][1]:
            q.append(i)
            for j in windows[i][0]:
                if not done[j]:
                    done[j] = True
                    doneCount += 1

    # Process the queue: When a window is "unstamped", it effectively turns its positions into '#'
    # and this might help other windows become unstampable.
    while q:
        win_idx = q.popleft()
        # Consider every position j in this window.
        for j in range(win_idx, win_idx + M):
            # For every window covering position j, remove j from its todo if present.
            for other_win in pos[j]:
                if j in windows[other_win][1]:
                    windows[other_win][1].remove(j)
                    if not windows[other_win][1]:
                        q.append(other_win)
                        for k in windows[other_win][0]:
                            if not done[k]:
                                done[k] = True
                                doneCount += 1

    # If all positions have been removed (turned to '#') in the reverse simulation,
    # then the forward stamping was possible.
    sys.stdout.write("Yes" if doneCount == N else "No")


if __name__ == '__main__':
    main()