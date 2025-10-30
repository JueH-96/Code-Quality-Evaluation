def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    pair_elements = []
    for i in range(N-1):
        if A[i] == A[i+1]:
            pair_elements.append(A[i])
    
    if not pair_elements:
        print(0)
        return
    
    last_occurrence = {}
    max_len = 0
    left = 0
    for right in range(len(pair_elements)):
        current = pair_elements[right]
        if current in last_occurrence and last_occurrence[current] >= left:
            left = last_occurrence[current] + 1
        last_occurrence[current] = right
        current_len = right - left + 1
        if current_len > max_len:
            max_len = current_len
    
    print(max_len * 2)

if __name__ == "__main__":
    main()