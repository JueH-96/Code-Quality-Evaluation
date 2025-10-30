def main():
    import sys
    input_line = sys.stdin.readline().strip()
    if not input_line:
        return
    N, X, Y, Z = map(int, input_line.split())
    
    # If X < Y, use inbound: train goes from X to Y, increasing order.
    if X < Y:
        if X < Z < Y:
            print("Yes")
        else:
            print("No")
    # Otherwise, if X > Y, use outbound: train goes from X to Y in decreasing order.
    else:
        if X > Z > Y:
            print("Yes")
        else:
            print("No")
            
if __name__ == '__main__':
    main()