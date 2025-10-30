def main():
    n = int(input().strip())
    s = input().strip()
    
    for i in range(len(s) - 2):
        if s[i:i+3] == "ABC":
            print(i + 1)
            return
            
    print(-1)

if __name__ == "__main__":
    main()