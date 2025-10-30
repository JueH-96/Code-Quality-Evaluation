def solve():
    s = input()
    n = len(s)
    char_indices = {}
    for i in range(n):
        char = s[i]
        if char not in char_indices:
            char_indices[char] = []
        char_indices[char].append(i)
    
    count = 0
    for char in char_indices:
        indices = char_indices[char]
        if len(indices) < 2:
            continue
        for i_index in range(len(indices)):
            for k_index in range(i_index + 1, len(indices)):
                i = indices[i_index]
                k = indices[k_index]
                if k > i + 1:
                    count += (k - i - 1)
                    
    print(count)

if __name__ == '__main__':
    solve()