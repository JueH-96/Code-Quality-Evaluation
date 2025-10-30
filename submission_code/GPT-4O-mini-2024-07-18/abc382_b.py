def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    # Read N and D
    N, D = map(int, data[0].split())
    # Read the string S
    S = list(data[1])
    
    # Process the cookies
    for _ in range(D):
        # Find the rightmost cookie
        for i in range(N - 1, -1, -1):
            if S[i] == '@':
                S[i] = '.'  # Eat the cookie
                break
    
    # Join the list back to a string and print the result
    print(''.join(S))

if __name__ == "__main__":
    main()