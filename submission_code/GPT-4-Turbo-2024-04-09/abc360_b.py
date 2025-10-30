def main():
    import sys
    input = sys.stdin.read
    S, T = input().split()
    
    len_S = len(S)
    len_T = len(T)
    
    # We need to find c and w such that 1 <= c <= w < len_S
    for w in range(1, len_S):
        for c in range(1, w + 1):
            # Collect the c-th characters from each valid chunk
            constructed_T = []
            for i in range(0, len_S, w):
                if i + c <= len_S:  # Ensure the chunk has at least c characters
                    constructed_T.append(S[i + c - 1])
            
            # Join the characters to form the new string
            if ''.join(constructed_T) == T:
                print("Yes")
                return
    
    print("No")

if __name__ == "__main__":
    main()