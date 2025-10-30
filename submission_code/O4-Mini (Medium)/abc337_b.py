def main():
    S = input().strip()
    n = len(S)
    
    i = 0
    # consume all leading 'A's
    while i < n and S[i] == 'A':
        i += 1
    # consume all following 'B's
    while i < n and S[i] == 'B':
        i += 1
    # consume all following 'C's
    while i < n and S[i] == 'C':
        i += 1
    
    # if we've consumed the entire string in A*B*C* fashion, it's valid
    if i == n:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()