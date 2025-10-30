def main():
    S = input().strip()
    if set(S) == set('ABC'):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()