import sys

def main():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    pos = [0] * (N + 1)  # pos[value] gives its index (0-based)
    for idx in range(N):
        num = A[idx]
        pos[num] = idx
    swaps = []
    for i in range(N):
        desired = i + 1
        if A[i] == desired:
            continue
        j = pos[desired]
        swaps.append((i + 1, j + 1))  # converting to 1-based for output
        # Swap elements in A
        A[i], A[j] = A[j], A[i]
        # Update positions
        pos[A[i]] = i
        pos[A[j]] = j
    print(len(swaps))
    for s in swaps:
        print(s[0], s[1])

if __name__ == "__main__":
    main()