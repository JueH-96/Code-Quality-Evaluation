def is_repdigit(n):
    s = str(n)
    return all(c == s[0] for c in s)

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    D = list(map(int, data[1:N+1]))
    count = 0
    for idx in range(N):
        month = idx + 1
        if is_repdigit(month):
            max_day = D[idx]
            for day in range(1, max_day + 1):
                if is_repdigit(day):
                    count +=1
    print(count)

if __name__ == "__main__":
    main()