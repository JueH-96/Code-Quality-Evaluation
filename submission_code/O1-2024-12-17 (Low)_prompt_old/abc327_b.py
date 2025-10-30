def solve():
    import sys
    B = int(sys.stdin.readline().strip())
    
    # A^A grows extremely quickly, so we only need to check up to a small limit.
    # Rough estimate: for B <= 10^18, A won't exceed ~60.
    limit = 60
    
    for A in range(1, limit + 1):
        val = pow(A, A)
        if val == B:
            print(A)
            return
        if val > B:
            break
    
    print(-1)

def main():
    solve()

if __name__ == "__main__":
    main()