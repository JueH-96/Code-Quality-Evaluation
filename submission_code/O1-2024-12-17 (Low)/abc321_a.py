def main():
    N = input().strip()
    # Check if the digits are strictly decreasing from left to right
    for i in range(len(N) - 1):
        if int(N[i]) <= int(N[i + 1]):
            print("No")
            return
    print("Yes")

main()