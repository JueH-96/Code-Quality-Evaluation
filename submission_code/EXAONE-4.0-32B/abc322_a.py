def main():
    import sys
    data = sys.stdin.read().splitlines()
    n = int(data[0].strip())
    s = data[1].strip()
    
    for i in range(len(s) - 2):
        if s[i:i+3] == "ABC":
            print(i + 1)
            return
    
    print(-1)

if __name__ == "__main__":
    main()