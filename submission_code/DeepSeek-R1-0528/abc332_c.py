def main():
    import sys
    data = sys.stdin.read().splitlines()
    n, m = map(int, data[0].split())
    s = data[1].strip()
    segments = s.split('0')
    ans = 0
    for seg in segments:
        if seg:
            total = len(seg)
            num2 = seg.count('2')
            req = max(num2, total - m)
            if req > ans:
                ans = req
    print(ans)

if __name__ == "__main__":
    main()