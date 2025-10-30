def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        print(0)
        return
    n, m = map(int, data[0].split())
    s = data[1].strip()
    
    segments = s.split('0')
    ans = 0
    for seg in segments:
        ones = seg.count('1')
        twos = seg.count('2')
        req = max(0, ones - m) + twos
        if req > ans:
            ans = req
            
    print(ans)

if __name__ == "__main__":
    main()