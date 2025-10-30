def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = int(data[1])
    B = int(data[2])
    D = list(map(int, data[3:3+N]))
    
    if N == 1:
        print("Yes")
        return
    
    # Compute differences between consecutive D_i
    diffs = [D[i] - D[i-1] for i in range(1, N)]
    
    # Check conditions
    for diff in diffs:
        if not (diff <= A - 1 or diff >= B + 1):
            print("No")
            return
    
    print("Yes")

if __name__ == "__main__":
    main()