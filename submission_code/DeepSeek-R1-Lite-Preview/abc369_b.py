def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    left_seq = []
    right_seq = []
    
    for i in range(1, N + 1):
        A, S = data[i].split()
        A = int(A)
        if S == 'L':
            left_seq.append(A)
        else:
            right_seq.append(A)
    
    def total_movement(seq):
        if not seq:
            return 0
        total = 0
        for i in range(1, len(seq)):
            total += abs(seq[i] - seq[i - 1])
        return total
    
    total_fatigue = total_movement(left_seq) + total_movement(right_seq)
    print(total_fatigue)

if __name__ == "__main__":
    main()