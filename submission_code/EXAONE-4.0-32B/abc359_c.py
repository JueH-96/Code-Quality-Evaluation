def main():
    import sys
    data = sys.stdin.read().split()
    sx = int(data[0])
    sy = int(data[1])
    tx = int(data[2])
    ty = int(data[3])
    
    part1 = abs((sx + sy) // 2 - (tx + ty) // 2)
    diff = (sy - sx) - (ty - tx)
    part2 = abs(diff) // 2
    
    ans = part1 + part2
    print(ans)

if __name__ == '__main__':
    main()