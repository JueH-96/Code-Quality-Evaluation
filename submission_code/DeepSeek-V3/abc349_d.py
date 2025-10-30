def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    L = int(data[0])
    R = int(data[1])
    
    sequences = []
    current = L
    while current < R:
        # Find the largest possible 2^i such that current is divisible by 2^i
        # and the next multiple of 2^i is <= R
        i = 0
        while True:
            next_step = current + (1 << i)
            if next_step > R:
                break
            if (current % (1 << i)) == 0:
                i += 1
            else:
                break
        i -= 1
        next_step = current + (1 << i)
        if next_step > R:
            next_step = R
        sequences.append((current, next_step))
        current = next_step
    
    print(len(sequences))
    for seq in sequences:
        print(seq[0], seq[1])

if __name__ == "__main__":
    main()