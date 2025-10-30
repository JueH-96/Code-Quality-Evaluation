def solve():
    s = input()
    counts = {}
    for char in s:
        counts[char] = counts.get(char, 0) + 1
    
    freq_counts = {}
    for count in counts.values():
        freq_counts[count] = freq_counts.get(count, 0) + 1
    
    for i in range(1, 101):
        if i not in freq_counts:
            freq_counts[i] = 0
    
    for count in freq_counts:
        if freq_counts[count] != 0 and freq_counts[count] != 2:
            print("No")
            return
    
    print("Yes")

solve()