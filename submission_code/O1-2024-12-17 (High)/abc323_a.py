def main():
    S = input().strip()
    # Check characters at even positions (1-based), i.e., indices 1,3,5,...,15 in 0-based indexing
    for i in range(1, 16, 2):
        if S[i] != '0':
            print("No")
            return
    print("Yes")

# Do not forget to call main() at the end
main()