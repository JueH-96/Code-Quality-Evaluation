def solve():
    s = input()
    letter_counts = {}
    for char in s:
        letter_counts[char] = letter_counts.get(char, 0) + 1
    
    frequency_counts = {}
    for char in letter_counts:
        frequency = letter_counts[char]
        frequency_counts[frequency] = frequency_counts.get(frequency, 0) + 1
        
    is_good_string = True
    for frequency in frequency_counts:
        count = frequency_counts[frequency]
        if count != 2:
            is_good_string = False
            break
            
    if is_good_string:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    solve()