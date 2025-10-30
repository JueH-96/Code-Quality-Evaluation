import sys

def main():
    S = sys.stdin.readline().strip()
    
    counts = []
    cnt = 0
    for ch in S[1:]:           # skip the very first '|'
        if ch == '-':
            cnt += 1
        else:                  # ch == '|'
            counts.append(cnt)
            cnt = 0

    print(' '.join(map(str, counts)))

if __name__ == "__main__":
    main()