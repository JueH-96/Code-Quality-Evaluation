def main():
    s = input().strip()
    idx_r = s.index('R')
    idx_m = s.index('M')
    if idx_r < idx_m:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()