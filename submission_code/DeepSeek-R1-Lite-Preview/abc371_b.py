def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    first_male = [False] * (N + 1)
    
    index = 2
    for _ in range(M):
        A_i = int(data[index])
        B_i = data[index + 1]
        index += 2
        if B_i == 'M' and not first_male[A_i]:
            print("Yes")
            first_male[A_i] = True
        else:
            print("No")

if __name__ == "__main__":
    main()