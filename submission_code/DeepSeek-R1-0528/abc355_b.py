def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    m = int(data[1])
    A = list(map(int, data[2:2+n]))
    B = list(map(int, data[2+n:2+n+m]))
    
    C = sorted(A + B)
    set_A = set(A)
    
    for i in range(len(C) - 1):
        if C[i] in set_A and C[i+1] in set_A:
            print("Yes")
            return
    
    print("No")

if __name__ == "__main__":
    main()