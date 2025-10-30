def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    n = int(input[ptr])
    ptr += 1
    P = list(map(int, input[ptr:ptr + n]))
    ptr += n
    Q = list(map(int, input[ptr:ptr + n]))
    
    bib_to_index = [0] * (n + 1)
    for i in range(n):
        bib = Q[i]
        bib_to_index[bib] = i
    
    result = []
    for i in range(1, n + 1):
        x = bib_to_index[i]
        target_person = P[x]
        s_i = Q[target_person - 1]
        result.append(str(s_i))
    
    print(' '.join(result))

if __name__ == "__main__":
    main()