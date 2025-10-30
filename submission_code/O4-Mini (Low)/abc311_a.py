def main():
    import sys
    
    data = sys.stdin.read().split()
    N = int(data[0])
    S = data[1]
    
    seen = set()
    for i, ch in enumerate(S):
        seen.add(ch)
        if len(seen) == 3:
            # i is 0-based, so the count of checked characters is i+1
            print(i + 1)
            return

if __name__ == "__main__":
    main()