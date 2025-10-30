def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    s = input_data[0]
    
    unique_substrings = set()
    for i in range(len(s)):
        for j in range(i+1, len(s)+1):
            unique_substrings.add(s[i:j])
            
    print(len(unique_substrings))

if __name__ == '__main__':
    main()