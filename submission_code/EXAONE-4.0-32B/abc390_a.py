def main():
    A = list(map(int, input().split()))
    if A == [1, 2, 3, 4, 5]:
        print("No")
    else:
        found = False
        for i in range(4):
            temp = A.copy()
            temp[i], temp[i+1] = temp[i+1], temp[i]
            if temp == [1, 2, 3, 4, 5]:
                found = True
                break
        print("Yes" if found else "No")

if __name__ == "__main__":
    main()