def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    a = list(map(int, data[1:1+n]))
    print("Yes" if len(set(a)) == 1 else "No")

if __name__ == "__main__":
    main()