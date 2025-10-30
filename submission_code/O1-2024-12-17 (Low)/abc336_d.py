def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Edge case: if N == 1, answer is 1 if A[0]>=1 (which problem guarantees)
    if N == 1:
        print(1)
        return

    # Linc[i] = length of the longest "1..Linc[i]" ascending-by-1 sequence ending at i
    Linc = [0]*N
    # Rinc[i] = length of the longest "1..Rinc[i]" ascending-by-1 sequence starting at i
    Rinc = [0]*N

    # Build Linc left-to-right
    # If A[i] >= Linc[i-1] + 1 then Linc[i] = Linc[i-1] + 1
    # else if A[i]>=1 we can at least start a new chain of length=1
    # else no valid "1..?" sequence for that position => 0
    if A[0] >= 1:
        Linc[0] = 1
    for i in range(1, N):
        if A[i] >= Linc[i-1] + 1:
            Linc[i] = Linc[i-1] + 1
        elif A[i] >= 1:
            Linc[i] = 1
        else:
            Linc[i] = 0

    # Build Rinc right-to-left
    if A[N-1] >= 1:
        Rinc[N-1] = 1
    for i in range(N-2, -1, -1):
        if A[i] >= Rinc[i+1] + 1:
            Rinc[i] = Rinc[i+1] + 1
        elif A[i] >= 1:
            Rinc[i] = 1
        else:
            Rinc[i] = 0

    # The pyramid's peak can be at i
    # The height (k) at i is min(Linc[i], Rinc[i]).
    # We want the maximum such k.
    ans = 0
    for i in range(N):
        peak = min(Linc[i], Rinc[i])
        if peak > ans:
            ans = peak

    print(ans)

# Don't forget to call main()
if __name__ == "__main__":
    main()