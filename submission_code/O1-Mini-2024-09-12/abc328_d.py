from collections import deque
import sys

def main():
    S = sys.stdin.read().strip()
    N = len(S)
    S = list(S)
    removed = [False] * N
    prev = [i-1 for i in range(N)]
    next = [i+1 for i in range(N)]
    next[N-1] = -1

    queue = deque()
    for i in range(N-2):
        if S[i] == 'A' and S[i+1] == 'B' and S[i+2] == 'C':
            queue.append(i)

    while queue:
        i = queue.popleft()
        if i < 0 or i+2 >= N:
            continue
        if removed[i] or removed[i+1] or removed[i+2]:
            continue
        # Remove ABC
        removed[i] = removed[i+1] = removed[i+2] = True
        # Link the previous and next characters
        pre = prev[i]
        nex = next[i+2]
        if pre != -1:
            next[pre] = nex
        if nex != -1:
            prev[nex] = pre
        # Check for new ABC starting from pre-2 to pre
        start = pre
        for j in range(start-2, start+1):
            if j < 0 or j+2 >= N:
                continue
            if not removed[j] and not removed[j+1] and not removed[j+2]:
                if S[j] == 'A' and S[j+1] == 'B' and S[j+2] == 'C':
                    queue.append(j)

    # Collect the remaining characters
    result = []
    i = 0
    while i < N:
        if not removed[i]:
            result.append(S[i])
        i += 1
    print(''.join(result))

if __name__ == "__main__":
    main()