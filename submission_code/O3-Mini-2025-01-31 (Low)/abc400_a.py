def main():
    import sys
    # Read the input value A from standard input
    data = sys.stdin.read().strip()
    if data == "":
        return
    try:
        A = int(data)
    except ValueError:
        return
    
    # Check if 400 is divisible by A
    if 400 % A == 0:
        B = 400 // A
        print(B)
    else:
        print(-1)

if __name__ == "__main__":
    main()