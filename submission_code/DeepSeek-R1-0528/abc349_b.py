def main():
    s = input().strip()
    from collections import defaultdict
    freq_per_char = defaultdict(int)
    for char in s:
        freq_per_char[char] += 1
        
    freq_count = defaultdict(int)
    for count in freq_per_char.values():
        freq_count[count] += 1
        
    if not freq_per_char:
        print("Yes")
        return
        
    max_freq = max(freq_per_char.values())
    valid = True
    for i in range(1, max_freq + 1):
        count_here = freq_count.get(i, 0)
        if count_here != 0 and count_here != 2:
            valid = False
            break
            
    print("Yes" if valid else "No")

if __name__ == '__main__':
    main()