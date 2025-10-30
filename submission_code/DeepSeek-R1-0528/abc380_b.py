def main():
    s = input().strip()
    s_mid = s[1:-1]
    parts = s_mid.split('|')
    ans = [str(len(part)) for part in parts]
    print(" ".join(ans))

if __name__ == "__main__":
    main()