def main():
    N = int(input())
    S = input()
    Q = int(input())
    
    for _ in range(Q):
        operation = input().split()
        t_i = int(operation[0])
        
        if t_i == 1:
            x_i = int(operation[1])
            c_i = operation[2]
            # Change the x_i-th character of S to c_i (1-indexed to 0-indexed)
            S = S[:x_i - 1] + c_i + S[x_i:]
        elif t_i == 2:
            # Convert all uppercase letters to lowercase
            S = S.lower()
        elif t_i == 3:
            # Convert all lowercase letters to uppercase
            S = S.upper()
    
    print(S)

if __name__ == "__main__":
    main()