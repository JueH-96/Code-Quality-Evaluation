def main():
    import sys
    N, *rest = map(int, sys.stdin.read().split())
    A = rest[:N]
    found = False
    for i in range(N):
        a = A[i]
        b = A[(i+1) % N]
        c = A[(i+2) % N]
        if a == 0 and b == 0 and c == 0:
            found = True
            break
    print("No" if found else "Yes")

if __name__ == "__main__":
    main()