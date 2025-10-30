import sys

def main() -> None:
    # Read input
    N_line = sys.stdin.readline()
    while N_line.strip() == "":
        N_line = sys.stdin.readline()
    N = int(N_line)
    S = sys.stdin.readline().strip()  # S should already be of length N
    
    # Find the first occurrence of "ABC"
    idx = S.find("ABC")   # 0-based; returns -1 if not found
    if idx == -1:
        print(-1)
    else:
        print(idx + 1)    # Convert to 1-based position

if __name__ == "__main__":
    main()