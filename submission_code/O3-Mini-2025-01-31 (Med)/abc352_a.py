def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N, X, Y, Z = map(int, input_data)

    # Determine train direction
    if X < Y:
        # Inbound train: stops at stations X+1, X+2, ..., Y
        if X < Z < Y:
            sys.stdout.write("Yes")
        else:
            sys.stdout.write("No")
    else:
        # Outbound train: stops at stations X-1, X-2, ..., Y
        if Y < Z < X:
            sys.stdout.write("Yes")
        else:
            sys.stdout.write("No")

if __name__ == '__main__':
    main()