def main():
    A = list(map(int, input().split()))
    target = [1, 2, 3, 4, 5]
    
    if A == target:
        print("No")
    else:
        for i in range(4):  # There are 4 adjacent pairs: (0,1), (1,2), (2,3), (3,4)
            B = A.copy()
            B[i], B[i+1] = B[i+1], B[i]
            if B == target:
                print("Yes")
                break
        else:
            print("No")

if __name__ == "__main__":
    main()