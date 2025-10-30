def main():
    S = input().strip()
    num_part = S[3:]
    n = int(num_part)
    if (1 <= n <= 315) or (317 <= n <= 349):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()