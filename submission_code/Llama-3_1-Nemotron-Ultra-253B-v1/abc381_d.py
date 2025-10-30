def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    blocks = []
    i = 0
    while i < N:
        if i + 1 < N and A[i] == A[i + 1]:
            current_block = []
            j = i
            while j + 1 < N and A[j] == A[j + 1]:
                current_block.append(A[j])
                j += 2
            blocks.append(current_block)
            i = j
        else:
            i += 1
    
    max_length = 0
    for block in blocks:
        start = 0
        last_occurrence = {}
        current_max = 0
        for end in range(len(block)):
            val = block[end]
            if val in last_occurrence and last_occurrence[val] >= start:
                start = last_occurrence[val] + 1
            last_occurrence[val] = end
            current_len = end - start + 1
            if current_len > current_max:
                current_max = current_len
        if current_max > max_length:
            max_length = current_max
    
    print(max_length * 2)

if __name__ == "__main__":
    main()