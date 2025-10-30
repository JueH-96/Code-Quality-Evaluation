def main():
    import sys

    data = sys.stdin.read().split()
    n = int(data[0])
    s = data[1]

    for i in range(n - 1):
        if (s[i] == 'a' and s[i+1] == 'b') or (s[i] == 'b' and s[i+1] == 'a'):
            print("Yes")
            return

    print("No")

if __name__ == "__main__":
    main()