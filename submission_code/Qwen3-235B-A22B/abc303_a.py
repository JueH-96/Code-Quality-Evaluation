def main():
    n = int(input())
    s = input()
    t = input()
    allowed = {('1', 'l'), ('l', '1'), ('0', 'o'), ('o', '0')}
    for a, b in zip(s, t):
        if a == b:
            continue
        if (a, b) in allowed:
            continue
        print("No")
        return
    print("Yes")

if __name__ == "__main__":
    main()