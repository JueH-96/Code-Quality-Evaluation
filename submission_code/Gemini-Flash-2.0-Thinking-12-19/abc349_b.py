def solve():
    s = input()
    char_counts = {}
    for char in s:
        char_counts[char] = char_counts.get(char, 0) + 1
    
    frequency_counts = {}
    for char in char_counts:
        frequency = char_counts[char]
        frequency_counts[frequency] = frequency_counts.get(frequency, 0) + 1
        
    is_good_string = True
    for frequency in frequency_counts:
        count = frequency_counts[frequency]
        if count != 0 and count != 2:
            is_good_string = False
            break
            
    if is_good_string:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    solve()