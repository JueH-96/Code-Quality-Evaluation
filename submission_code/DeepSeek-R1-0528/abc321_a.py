def main():
    n_str = input().strip()
    if len(n_str) == 1:
        print("Yes")
        return
        
    for i in range(len(n_str) - 1):
        if n_str[i] <= n_str[i+1]:
            print("No")
            return
            
    print("Yes")

if __name__ == "__main__":
    main()