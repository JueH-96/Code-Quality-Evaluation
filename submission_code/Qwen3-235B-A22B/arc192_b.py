def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    if 1 in A:
        if N % 2 == 1:
            print("Fennec")
        else:
            print("Snuke")
    else:
        total = sum(A)
        if total % 2 == 1:
            print("Fennec")
        else:
            print("Snuke")

if __name__ == "__main__":
    main()