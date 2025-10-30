def main():
    data = input().split()
    if len(data) < 2:
        print("No")
        return
        
    S = data[0]
    T = data[1]
    
    if S == "AtCoder" and T == "Land":
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()