def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    if all(a == 1 for a in A):
        print("Yes")
        return
    
    # Check for three consecutive zeros in circular array
    A_extended = A * 2  # To handle circularity
    found = False
    for i in range(N):
        if A_extended[i] == 0 and A_extended[i+1] == 0 and A_extended[i+2] == 0:
            found = True
            break
    if found:
        print("No")
    else:
        print("Yes")

if __name__ == "__main__":
    main()