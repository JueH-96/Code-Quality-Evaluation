def main():
    S, T = input().split()
    n = len(S)
    
    for w in range(1, n):
        chunks = []
        start = 0
        while start < n:
            end = start + w
            chunk = S[start:end]
            chunks.append(chunk)
            start = end
        
        for c in range(1, w + 1):
            extracted_chars = []
            for chunk in chunks:
                if len(chunk) >= c:
                    extracted_chars.append(chunk[c - 1])
            candidate = ''.join(extracted_chars)
            if candidate == T:
                print("Yes")
                return
    
    print("No")

if __name__ == '__main__':
    main()