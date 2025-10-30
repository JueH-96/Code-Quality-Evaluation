def main():
    S = input().strip()
    n = len(S)
    if n == 0:
        print(0)
        return
        
    max_len = 1
    
    for i in range(n):
        # Check for odd length palindromes with center i
        low = i
        high = i
        while low >= 0 and high < n and S[low] == S[high]:
            length = high - low + 1
            if length > max_len:
                max_len = length
            low -= 1
            high += 1
        
        # Check for even length palindromes with center between i and i+1
        low = i
        high = i + 1
        while low >= 0 and high < n and S[low] == S[high]:
            length = high - low + 1
            if length > max_len:
                max_len = length
            low -= 1
            high += 1
            
    print(max_len)

if __name__ == "__main__":
    main()