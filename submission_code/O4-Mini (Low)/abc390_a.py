def main():
    A = list(map(int, input().split()))
    target = [1, 2, 3, 4, 5]
    
    # Try every possible single adjacent swap
    for i in range(4):
        B = A[:]               # copy the list
        B[i], B[i+1] = B[i+1], B[i]
        if B == target:
            print("Yes")
            return
    
    # If no single adjacent swap yields the sorted list
    print("No")

if __name__ == "__main__":
    main()