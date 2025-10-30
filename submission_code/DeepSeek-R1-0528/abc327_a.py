def main():
    n = int(input().strip())
    s = input().strip()
    
    for i in range(n - 1):
        substring = s[i:i+2]
        if substring == "ab" or substring == "ba":
            print("Yes")
            return
            
    print("No")

if __name__ == "__main__":
    main()