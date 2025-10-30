def main():
    s = input().strip()
    groups = []
    prev = ''
    for char in s:
        if char != prev:
            groups.append(char)
            prev = char
            
    mapping = {'A': 0, 'B': 1, 'C': 2}
    last = -1
    for char in groups:
        num = mapping[char]
        if num < last:
            print("No")
            return
        last = num
        
    print("Yes")

if __name__ == '__main__':
    main()