def main():
    n_val = int(input().strip())
    s = input().strip()
    
    max_run = {}
    i = 0
    n = len(s)
    
    while i < n:
        j = i
        while j < n and s[j] == s[i]:
            j += 1
        length = j - i
        char = s[i]
        if char in max_run:
            if length > max_run[char]:
                max_run[char] = length
        else:
            max_run[char] = length
        i = j
    
    total = sum(max_run.values())
    print(total)

if __name__ == "__main__":
    main()