def main():
    N = int(input().strip())
    S = input().strip()
    
    for i in range(N - 1):
        if (S[i] == 'a' and S[i+1] == 'b') or (S[i] == 'b' and S[i+1] == 'a'):
            print("Yes")
            return
    print("No")

# Do not forget to call main function
main()