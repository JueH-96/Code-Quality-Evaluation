def main():
    import sys
    input = sys.stdin.readline

    # Read N, M
    N, M = map(int, input().split())
    # Read sequence A and B
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))

    # Create a set for quick membership checks in A
    setA = set(A)

    # Merge and sort the combined sequence
    C = sorted(A + B)

    # Scan through adjacent pairs in C
    for i in range(len(C) - 1):
        # If both adjacent elements come from A, print "Yes" and exit
        if C[i] in setA and C[i+1] in setA:
            print("Yes")
            return

    # If no such adjacent A-elements were found, print "No"
    print("No")

if __name__ == "__main__":
    main()