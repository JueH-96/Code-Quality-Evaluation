def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    m = int(data[1])
    l = list(map(int, data[2:2+n]))
    
    max_l = max(l)
    sum_l = sum(l) + (n - 1)  # sum of all words plus spaces for one line
    
    low = max_l
    high = sum_l
    
    def is_possible(w):
        lines = 1
        current = 0
        for word in l:
            if current == 0:
                current = word
            else:
                if current + 1 + word <= w:
                    current += 1 + word
                else:
                    lines += 1
                    current = word
                    if lines > m:
                        return False
        return lines <= m
    
    while low < high:
        mid = (low + high) // 2
        if is_possible(mid):
            high = mid
        else:
            low = mid + 1
    
    print(low)

if __name__ == "__main__":
    main()