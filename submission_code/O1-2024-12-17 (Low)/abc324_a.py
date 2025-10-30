def main():
    N = int(input().strip())
    A = list(map(int, input().split()))
    
    if all(x == A[0] for x in A):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()