def main():
    n = int(input().strip())
    arr = [input().strip() for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            s = arr[i] + arr[j]
            if s == s[::-1]:
                print("Yes")
                return
                
    print("No")

if __name__ == "__main__":
    main()