def solve():
    import sys
    S = sys.stdin.readline().strip()
    
    # Since S is guaranteed to end with '2023', we just replace the last character with '4'
    answer = S[:-1] + '4'
    
    print(answer)

def main():
    solve()

if __name__ == "__main__":
    main()