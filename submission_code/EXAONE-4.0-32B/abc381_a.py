def main():
    import sys
    data = sys.stdin.read().splitlines()
    n = int(data[0].strip())
    s = data[1].strip()
    
    if n % 2 == 0:
        print("No")
        return
        
    mid = (n - 1) // 2
    if s[mid] != '/':
        print("No")
        return
        
    for i in range(mid):
        if s[i] != '1':
            print("No")
            return
            
    for i in range(mid + 1, n):
        if s[i] != '2':
            print("No")
            return
            
    print("Yes")

if __name__ == "__main__":
    main()