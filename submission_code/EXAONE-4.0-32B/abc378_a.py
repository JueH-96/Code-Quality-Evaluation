from collections import Counter

def main():
    data = list(map(int, input().split()))
    cnt = Counter(data)
    freqs = sorted(cnt.values(), reverse=True)
    
    if freqs[0] == 4:
        print(2)
    elif freqs[0] == 3:
        print(1)
    elif freqs[0] == 2:
        if len(freqs) > 1 and freqs[1] == 2:
            print(2)
        else:
            print(1)
    else:
        print(0)

if __name__ == "__main__":
    main()