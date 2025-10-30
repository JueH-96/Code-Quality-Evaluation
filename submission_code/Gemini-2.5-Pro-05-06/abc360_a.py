def solve():
    s = input()
    
    # Find the index of 'R' (rice)
    index_r = -1
    for i in range(len(s)):
        if s[i] == 'R':
            index_r = i
            break
            
    # Find the index of 'M' (miso soup)
    index_m = -1
    for i in range(len(s)):
        if s[i] == 'M':
            index_m = i
            break
            
    # An alternative and more Pythonic way to find indices:
    # index_r = s.find('R')
    # index_m = s.find('M')
    
    # Determine if rice is to the left of miso soup
    if index_r < index_m:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()