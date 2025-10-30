def main():
    import sys
    from itertools import permutations

    data = sys.stdin.read().strip().split()
    A = list(map(int, data))

    for B in permutations(A, 3):
        if B[0] * B[1] == B[2]:
            print("Yes")
            return
    print("No")

if __name__ == "__main__":
    main()