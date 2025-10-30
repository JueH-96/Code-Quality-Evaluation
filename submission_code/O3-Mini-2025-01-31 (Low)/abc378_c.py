def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    A = list(map(int, input_data[1:n+1]))
    
    last_occurrence = {}
    B = []
    
    for i, a in enumerate(A):
        if a in last_occurrence:
            # Already seen, assign last occurrence (converting from 0-indexed to 1-indexed)
            B.append(last_occurrence[a] + 1)
        else:
            B.append(-1)
        last_occurrence[a] = i
    sys.stdout.write(" ".join(map(str, B)))
    
if __name__ == '__main__':
    main()