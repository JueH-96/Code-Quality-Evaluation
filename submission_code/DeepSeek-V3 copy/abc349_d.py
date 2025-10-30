def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    L = int(data[0])
    R = int(data[1])
    
    result = []
    current = L
    while current < R:
        # Find the largest possible 2^i such that current is divisible by 2^i
        # and current + 2^i <= R
        i = 0
        while True:
            next_step = 1 << i
            if current % next_step != 0:
                break
            if current + next_step > R:
                break
            i += 1
        i -= 1
        next_step = 1 << i
        end = current + next_step
        result.append((current, end))
        current = end
    
    print(len(result))
    for l, r in result:
        print(l, r)

if __name__ == "__main__":
    main()