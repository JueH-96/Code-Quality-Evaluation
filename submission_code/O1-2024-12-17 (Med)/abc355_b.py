def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    A = list(map(int, data[2:2+N]))
    B = list(map(int, data[2+N:2+N+M]))
    
    set_A = set(A)
    C = sorted(A + B)
    
    for i in range(len(C) - 1):
        if C[i] in set_A and C[i+1] in set_A:
            print("Yes")
            return
    print("No")

main()