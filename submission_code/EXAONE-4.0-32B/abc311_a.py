def main():
    n = int(input().strip())
    s = input().strip()
    
    seen = set()
    for idx, char in enumerate(s):
        seen.add(char)
        if len(seen) == 3:
            print(idx + 1)
            return

if __name__ == "__main__":
    main()