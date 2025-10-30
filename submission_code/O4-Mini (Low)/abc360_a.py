def main():
    import sys
    S = sys.stdin.readline().strip()
    # Find positions of rice (R) and miso soup (M)
    pos_r = S.index('R')
    pos_m = S.index('M')
    # Check if rice is to the left of miso soup
    if pos_r < pos_m:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()