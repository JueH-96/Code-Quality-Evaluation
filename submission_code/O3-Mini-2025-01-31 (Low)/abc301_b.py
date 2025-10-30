def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    n = int(data[0])
    seq = list(map(int, data[1:]))
    
    # While there's any adjacent pair whose difference is not 1, perform insertion.
    while True:
        inserted = False
        for i in range(len(seq) - 1):
            if abs(seq[i+1] - seq[i]) != 1:
                # If there's a gap, generate the missing numbers based on order.
                if seq[i] < seq[i+1]:
                    new_numbers = list(range(seq[i] + 1, seq[i+1]))
                else:
                    new_numbers = list(range(seq[i] - 1, seq[i+1], -1))
                # Insert new_numbers between seq[i] and seq[i+1]
                seq = seq[:i+1] + new_numbers + seq[i+1:]
                inserted = True
                break
        if not inserted:
            break

    print(" ".join(map(str, seq)))

if __name__ == '__main__':
    main()