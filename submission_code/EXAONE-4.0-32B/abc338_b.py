def main():
    S = input().strip()
    freq = [0] * 26
    for ch in S:
        idx = ord(ch) - ord('a')
        freq[idx] += 1
        
    max_count = 0
    result_char = ''
    for i in range(26):
        if freq[i] > max_count:
            max_count = freq[i]
            result_char = chr(ord('a') + i)
            
    print(result_char)

if __name__ == '__main__':
    main()