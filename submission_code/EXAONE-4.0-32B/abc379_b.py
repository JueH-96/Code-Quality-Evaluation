def main():
    import sys
    data = sys.stdin.read().splitlines()
    n, k = map(int, data[0].split())
    s = data[1].strip()
    
    ans = 0
    current_run = 0
    for char in s:
        if char == 'O':
            current_run += 1
        else:
            ans += current_run // k
            current_run = 0
    ans += current_run // k
    print(ans)

if __name__ == "__main__":
    main()