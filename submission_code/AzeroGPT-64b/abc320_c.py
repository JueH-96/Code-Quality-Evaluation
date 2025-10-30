from itertools import chain

def main():
    M = int(input())
    reels = [input() for _ in range(3)]
    symbols = list(set(chain(*reels)))
    inf = 999999999

    if len(symbols) == 1:
        print(0)
        return

    options = []
    for symbol in symbols:
        secs = []
        for reel in reels:
            for i, char in enumerate(reel):
                if char == symbol:
                    secs.append(M - i if i > M // 2 else i)
        if len(secs) < 3:
            continue
        options.append(max(secs))

    if options:
        print(min(options))
    else:
        print(-1)

if __name__ == "__main__":
    main()