def solve():
    s = input()
    counts = {}
    for char in s:
        counts[char] = counts.get(char, 0) + 1
    
    freq_counts = {}
    for char in counts:
        freq = counts[char]
        freq_counts[freq] = freq_counts.get(freq, 0) + 1
    
    for i in range(1, len(s) + 1):
        if i in freq_counts:
            if freq_counts[i] != 0 and freq_counts[i] != 2:
                print("No")
                return
    
    print("Yes")

solve()