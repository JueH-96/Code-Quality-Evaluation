def main():
    import sys
    input = sys.stdin.readline

    n = int(input().strip())
    unique_sticks = set()
    
    for _ in range(n):
        s = input().strip()
        # Get the canonical form by taking the lexicographically smallest string of s and its reverse.
        canon = min(s, s[::-1])
        unique_sticks.add(canon)
    
    print(len(unique_sticks))
    
if __name__ == "__main__":
    main()