def main():
    import sys
    data = sys.stdin.read().split()
    # first element is N, then following are list elements
    if not data:
        return
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    found = False
    for i in range(N - 2):
        if A[i] == A[i+1] == A[i+2]:
            found = True
            break
    print("Yes" if found else "No")

if __name__ == "__main__":
    main()