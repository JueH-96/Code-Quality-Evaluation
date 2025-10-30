def main():
    import sys
    data = sys.stdin.read().split()
    if len(data) < 2:
        return
    S, T = data[0], data[1]
    if S == "AtCoder" and T == "Land":
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    main()