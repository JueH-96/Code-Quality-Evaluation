def main():
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    N, X, Y, Z = map(int, data)
    
    if X < Y:
        # Inbound train from X to Y
        if X <= Z <= Y:
            print("Yes")
        else:
            print("No")
    else:
        # Outbound train from X to Y
        if Y <= Z <= X:
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()