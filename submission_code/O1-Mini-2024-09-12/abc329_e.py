# YOUR CODE HERE
def main():
    import sys

    N, M = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    # Find all positions where T matches S starting at pos
    matches = []
    for i in range(N - M +1):
        if S[i:i+M] == T:
            matches.append(i)

    if not matches:
        print("No")
        return

    current = 0
    idx = 0
    last = len(matches)
    while current < N:
        # Find the farthest match that starts at or before current
        best = -1
        while idx < last and matches[idx] <= current:
            best = matches[idx] + M
            idx +=1
        if best == -1:
            print("No")
            return
        current = best
    print("Yes")

if __name__ == "__main__":
    main()