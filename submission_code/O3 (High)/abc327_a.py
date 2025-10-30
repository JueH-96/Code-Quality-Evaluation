def main():
    import sys
    
    # Read input
    N_line = sys.stdin.readline()
    while N_line.strip() == '':
        N_line = sys.stdin.readline()
    N = int(N_line)
    S = sys.stdin.readline().strip()
    
    # Check adjacent characters
    for i in range(N - 1):
        pair = S[i:i+2]
        if pair in ("ab", "ba"):
            print("Yes")
            return
    print("No")

if __name__ == "__main__":
    main()