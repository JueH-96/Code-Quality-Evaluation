def main():
    n = int(input().strip())
    seq = list(map(int, input().split()))
    
    while True:
        inserted = False
        for i in range(len(seq) - 1):
            if abs(seq[i] - seq[i+1]) != 1:
                if seq[i] < seq[i+1]:
                    to_insert = list(range(seq[i] + 1, seq[i+1]))
                else:
                    to_insert = list(range(seq[i] - 1, seq[i+1], -1))
                seq = seq[:i+1] + to_insert + seq[i+1:]
                inserted = True
                break
        if not inserted:
            break
    print(" ".join(map(str, seq)))

if __name__ == "__main__":
    main()