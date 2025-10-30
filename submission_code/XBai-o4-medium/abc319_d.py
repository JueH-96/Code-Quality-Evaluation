def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    m = int(input[1])
    l = list(map(int, input[2:2+n]))
    
    def is_possible(W):
        lines = 1
        current = 0
        for word in l:
            if current == 0:
                current = word
            else:
                if current + 1 + word <= W:
                    current += 1 + word
                else:
                    lines += 1
                    current = word
                    if lines > m:
                        return False
        return lines <= m
    
    low = max(l)
    high = sum(l) + (n - 1)
    ans = high
    while low <= high:
        mid = (low + high) // 2
        if is_possible(mid):
            ans = mid
            high = mid - 1
        else:
            low = mid + 1
    print(ans)

if __name__ == "__main__":
    main()