import sys
import threading

def main():
    import sys
    import heapq

    S = list(sys.stdin.readline().rstrip())
    n = len(S)
    # active[i] is True if S[i:i+2] == "WA"
    active = [False] * (n - 1)
    heap = []
    for i in range(n - 1):
        if S[i] == 'W' and S[i+1] == 'A':
            active[i] = True
            heapq.heappush(heap, i)

    while heap:
        i = heapq.heappop(heap)
        # if it's no longer "WA", skip
        if not active[i]:
            continue
        # perform the replacement at positions i, i+1
        # WA -> AC
        S[i]   = 'A'
        S[i+1] = 'C'
        active[i] = False

        # check neighbors where new "WA" might form or break:
        for j in (i-1, i, i+1):
            if 0 <= j < n-1:
                is_wa = (S[j] == 'W' and S[j+1] == 'A')
                if is_wa and not active[j]:
                    active[j] = True
                    heapq.heappush(heap, j)
                elif not is_wa and active[j]:
                    active[j] = False

    sys.stdout.write(''.join(S))

if __name__ == "__main__":
    main()