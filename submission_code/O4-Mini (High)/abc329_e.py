import sys
from collections import deque

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    S = next(it)
    T = next(it)
    # Number of possible stamp positions
    windowsCount = N - M + 1
    # stamped[i] = whether we've "erased" (reverse-stamped) at position i
    stamped = [False] * windowsCount
    # mismatchMask[i]: bitmask of positions j in [0..M-1] where S[i+j] != T[j]
    mismatchMask = [0] * windowsCount
    # todoCount[i]: how many mismatches remain for window i
    todoCount = [0] * windowsCount

    queue = deque()
    # Initialize masks and the queue of windows that match exactly (mask==0)
    for i in range(windowsCount):
        mask = 0
        for j in range(M):
            if S[i+j] != T[j]:
                mask |= 1 << j
        mismatchMask[i] = mask
        cnt = mask.bit_count()
        todoCount[i] = cnt
        if cnt == 0:
            queue.append(i)

    # erased[p] = whether position p in S has been turned into '#'
    erased = [False] * N
    replacedCount = 0

    # Reverse stamping: erase windows when all mismatches there have already been erased
    while queue:
        i = queue.popleft()
        if stamped[i]:
            continue
        stamped[i] = True
        # Erase positions p = i..i+M-1 if not already erased
        for j in range(M):
            p = i + j
            if not erased[p]:
                erased[p] = True
                replacedCount += 1
                # For each window k that covers p, see if it had a mismatch at p;
                # if so, decrement its todoCount and maybe enqueue it.
                start_k = p - M + 1
                if start_k < 0:
                    start_k = 0
                end_k = p
                if end_k > windowsCount - 1:
                    end_k = windowsCount - 1
                for k in range(start_k, end_k + 1):
                    j2 = p - k
                    # If window k originally mismatched at offset j2
                    if (mismatchMask[k] >> j2) & 1:
                        todoCount[k] -= 1
                        mismatchMask[k] ^= (1 << j2)
                        if todoCount[k] == 0:
                            queue.append(k)

    # If we've erased all N positions, we can build S by stamping forward
    print("Yes" if replacedCount == N else "No")

if __name__ == "__main__":
    main()